"""
Find the maximum micro batch size for each setting.

python -m memory.time_elapse_find_batch_size

JAX:
python -m memory.jax_dp --batch_size 5 --seq_len 100 --num_updates 2 --model_name_or_path gpt2
"""
import os

import fire

from lxuechen_utils import utils


def _run_command(
    mode,
    micro_batch_size,
    gradient_accumulation_steps,
    seq_len,
    model_name_or_path,
    num_updates,
    out_path=None,
    cache_dir="/nlp/scr/lxuechen/prefixtune/memory/cache",
):
    if mode == "jax":
        out = os.system(
            f"python -m memory.jax_dp_grad_accumulation "
            f"--batch_size {micro_batch_size} "
            f"--gradient_accumulation_steps {gradient_accumulation_steps} "
            f"--seq_len {seq_len} "
            f"--model_name_or_path {model_name_or_path} "
            f"--out_path {out_path} "
            f"--num_updates {num_updates}"
            f"--cache_dir {cache_dir}"
        )
    else:
        out = os.system(
            f"python -m memory.torch_dp "
            f"--mode {mode} "
            f"--batch_size {micro_batch_size} "
            f"--gradient_accumulation_steps {gradient_accumulation_steps} "
            f"--seq_len {seq_len} "
            f"--model_name_or_path {model_name_or_path} "
            f"--out_path {out_path} "
            f"--num_updates {num_updates} "
            f"--cache_dir {cache_dir}"
        )
    return out


def main(
    seq_lens=(100,),
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    modes=("nonprivate", "vanilla", "layer_by_layer", "ghost", "jax"),

    num_updates=2,
    gradient_accumulation_steps=2,
    min_micro_batch_size=1,
    max_micro_batch_size=100,
    threshold=1,  # A loose threshold for binary search.
):
    config2bsz = dict()
    for seq_len in seq_lens:
        for model_name_or_path in model_name_or_paths:
            for mode in modes:

                # Check first you can fit the minimum number of examples.
                print(f"model_name_or_path: {model_name_or_path}, mode: {mode}")

                out = _run_command(
                    mode,
                    min_micro_batch_size,
                    gradient_accumulation_steps,
                    seq_len,
                    model_name_or_path,
                    num_updates,
                )
                if out != 0:  # Failed -- cannot even fit a single example.
                    config2bsz[(model_name_or_path, mode, seq_len)] = 0
                    print("Can't even fit a single example, skipping...")
                    continue

                # Run binary search to get maximum batch size.
                while (max_micro_batch_size - min_micro_batch_size) > threshold:
                    mid_micro_batch_size = (min_micro_batch_size + max_micro_batch_size) // 2
                    print(f"Trying micro batch size: {mid_micro_batch_size}")

                    out = _run_command(
                        mode=mode,
                        micro_batch_size=mid_micro_batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        seq_len=seq_len,
                        model_name_or_path=model_name_or_path,
                        num_updates=num_updates,
                    )
                    if out != 0:  # Did fit.
                        max_micro_batch_size = mid_micro_batch_size
                        print(f"Micro batch size failed: {mid_micro_batch_size}")
                    else:  # Fit, try harder.
                        min_micro_batch_size = mid_micro_batch_size
                        print(f"Micro batch size succeeded: {mid_micro_batch_size}")

                # Take the left end as the maximum batch size.
                config2bsz[(model_name_or_path, mode, seq_len)] = min_micro_batch_size

    config_dir = f"/nlp/scr/lxuechen/prefixtune/memory/time_elapse_micro_batch_size.json"
    utils.jdump(config2bsz, config_dir)


if __name__ == '__main__':
    fire.Fire(main)
