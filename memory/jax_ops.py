import flax.linen as nn
import jax
import jax.numpy as jnp


def cross_entropy(logits, labels):
    """
    Computes the cross entroy loss (on logits).

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].
        ignore_index (int): Value of label to ignore for loss computation.

    Returns:
        (tensor): Cross entroy loss.
    """
    batch_size, num_classes = logits.shape
    logits = nn.log_softmax(logits)

    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    mult = one_hot_labels * logits
    # Inner sum over vocab; outer sum over sequence length.
    return -jnp.sum(jnp.sum(mult, axis=-1))
