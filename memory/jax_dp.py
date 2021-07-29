import functools
import itertools
import os
import time

import fire
import flaxmodels as fm
import jax
from jax.experimental import optimizers
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_multimap, tree_unflatten
import numpy as np
import tqdm

from lxuechen_utils import utils
from . import jax_ops

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def clipped_grad(model, loss, params, l2_norm_clip, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = jax.grad(functools.partial(loss, model))(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads)


def private_grad(model, loss, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads = jax.vmap(functools.partial(clipped_grad, model, loss), (None, None, 0))(
        params, l2_norm_clip, batch
    )
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = jax.random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * jax.random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)


def private_grad_no_vmap(model, loss, params, batch, rng, l2_norm_clip, noise_multiplier,
                         batch_size):
    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads = tree_multimap(
        lambda *xs: jnp.stack(xs),
        *(clipped_grad(model, loss, params, l2_norm_clip, eg) for eg in zip(*batch)))

    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = jax.random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * jax.random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)


def make_data(seq_len=10, batch_size=16):
    return (np.random.randint(low=0, high=100, size=(batch_size, seq_len)),)


def next_token_loss(model, params, batch):
    input_ids = batch[0]
    outputs = model.apply(params, input_ids=input_ids)
    lm_logits = outputs["logits"]

    shift_logits = lm_logits[..., :-1, :]
    shift_labels = input_ids[..., 1:]
    loss = jax_ops.cross_entropy(
        jnp.reshape(shift_logits, (-1, shift_logits.shape[-1])), jnp.reshape(shift_labels, (-1))
    )
    return loss


def main(
    seq_len=100,
    batch_size=5,
    num_warmups=3,
    num_updates=100,
    model_name_or_path="gpt2",
    seed=42,

    learning_rate=1e-4,
    l2_norm_clip=0.1,
    noise_multiplier=1.0,

    loss=next_token_loss,
    grad_fn=private_grad,

    no_jit=False,
    no_vmap=False,

    out_path=None,
):
    rng = jax.random.PRNGKey(seed)
    batch = make_data(seq_len, batch_size)

    model = fm.gpt2.GPT2LMHeadModel(pretrained=model_name_or_path)
    params = model.init(rng, input_ids=batch[0])
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    grad_fn = private_grad_no_vmap if no_vmap else private_grad

    def private_update(rng, i, opt_state, batch):
        params = get_params(opt_state)
        rng = jax.random.fold_in(rng, i)  # get new key for new random numbers
        return opt_update(
            i,
            grad_fn(
                model, loss, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size
            ),
            opt_state
        )

    opt_state = opt_init(params)
    itercount = itertools.count()
    train_fn = private_update

    if not no_jit:
        train_fn = jax.jit(train_fn)

    for _ in tqdm.tqdm(range(num_warmups), desc="warmup"):
        opt_state = train_fn(
            rng,
            next(itercount),
            opt_state,
            batch,
        )

    now = time.perf_counter()
    for _ in tqdm.tqdm(range(num_updates), desc='training'):
        opt_state = train_fn(
            rng,
            next(itercount),
            opt_state,
            batch,
        )
    time_elapse = time.perf_counter() - now
    print(f'{num_updates} updates spent {time_elapse:.4f} seconds')

    if out_path is not None:
        utils.jdump(
            {
                "time_elapse": time_elapse,
                "num_updates": num_updates,
                "num_warmups": num_warmups,
                "model_name_or_path": model_name_or_path,
                "seq_len": seq_len,
                "batch_size": batch_size,
            },
            out_path
        )


if __name__ == "__main__":
    # python -m memory.jax_dp     --no_jit True --no_vmap True
    # python -m memory.jax_dp     --no_jit False --no_vmap False
    fire.Fire(main)
