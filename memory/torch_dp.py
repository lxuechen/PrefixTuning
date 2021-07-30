import time

import fire
import torch
import tqdm
import transformers

from lxuechen_utils import utils


def make_data(seq_len=10, batch_size=16, device=None):
    return (torch.randint(low=0, high=100, size=(batch_size, seq_len), device=device),)


def train_step(model, optimizer, criterion, batch, mode, step=1, gradient_accumulation_steps=1):
    input_ids, = batch
    outputs = model(input_ids=input_ids, return_dict=True)
    lm_logits = outputs[0]
    shift_logits = lm_logits[..., :-1, :].permute(0, 2, 1)  # (batch_size, vocab_size, seq_len).
    shift_labels = input_ids[..., 1:]

    loss = criterion(shift_logits, shift_labels)
    loss = loss.sum(dim=1)

    if mode in ("vanilla", "per_layer", "nonprivate"):
        first_loss = loss.mean(dim=0)
        first_loss.backward()
    else:
        privacy_engine = optimizer.privacy_engine

        privacy_engine.set_hooks_mode(mode="norm")
        first_loss = loss.mean(dim=0)
        first_loss.backward(retain_graph=True)  # Must retain graph; otherwise dropout could be different.

        privacy_engine.set_hooks_mode(mode="grad")
        coef_sample = privacy_engine.get_coef_sample()
        # Sum here, since division is taken in `step`.
        second_loss = (coef_sample * loss).sum(dim=0)  # This is usual backprop, so take sum.
        second_loss.backward()

    if step % gradient_accumulation_steps == 0:
        optimizer.step()
        model.zero_grad()
    else:
        if hasattr(optimizer, 'virtual_step'):
            optimizer.virtual_step()


def main(
    mode="ghost",
    seq_len=100,
    batch_size=5,
    gradient_accumulation_steps=1,
    num_warmups=3,
    model_name_or_path="gpt2",
    num_updates=100,
    seed=42,

    learning_rate=1e-4,
    l2_norm_clip=0.1,
    noise_multiplier=1.0,

    out_path=None,
):
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = make_data(seq_len, batch_size, device)
    config = transformers.GPT2Config.from_pretrained(model_name_or_path)
    config.tie_word_embeddings = False
    config.copy_word_embeddings = True
    model = transformers.GPT2LMHeadModel.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, config=config,
    ).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if mode != "nonprivate":
        if mode == "vanilla":
            import privacy_utils
            cls = privacy_utils.privacy_engine.PrivacyEngine
        elif mode == "layer_by_layer":
            from experimental.privacy_utils.privacy_engine import EfficientPrivacyEngine
            cls = EfficientPrivacyEngine
        elif mode == "ghost":
            from experimental3.privacy_utils.privacy_engine import EfficientPrivacyEngine3
            cls = EfficientPrivacyEngine3
        elif mode == "per_layer":
            from experimental2.privacy_utils.privacy_engine import EfficientPrivacyEngine2
            cls = EfficientPrivacyEngine2
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Estimating the cost -- these parameters don't matter at all.
        actual_batch_size = gradient_accumulation_steps * batch_size
        privacy_engine = cls(
            module=model,
            batch_size=actual_batch_size,
            sample_size=100000,
            gradient_accumulation_steps=gradient_accumulation_steps,
            epochs=5,
            max_grad_norm=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            loss_reduction="mean",
            batch_first=True,
        )
        privacy_engine.attach(optimizer)

    model.train()
    model.zero_grad()

    for _ in tqdm.tqdm(range(num_warmups), desc="warmup"):
        train_step(model, optimizer, criterion, batch, mode)  # step, gradient_accumulation_steps don't matter here.

    print(f'Start training gradient_accumulation_steps: {gradient_accumulation_steps}')
    now = time.perf_counter()
    for step in tqdm.tqdm(range(1, num_updates + 1), desc="update"):
        train_step(model, optimizer, criterion, batch, mode, step, gradient_accumulation_steps)
    time_elapse = time.perf_counter() - now
    print(f'{num_updates} updates took {time_elapse:.4f} seconds')

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
    # python -m memory.torch_dp --mode "vanilla" --batch_size 3 --model_name_or_path "gpt2-large"
    # python -m memory.torch_dp --mode "ghost"
    # python -m memory.torch_dp --mode "ghost" --model_name_or_path "gpt2-large"
    fire.Fire(main)
