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
        command = f'''python -m memory.jax_dp_grad_accumulation \
            --batch_size {micro_batch_size} \
            --gradient_accumulation_steps {gradient_accumulation_steps} \
            --seq_len {seq_len} \
            --model_name_or_path {model_name_or_path} \
            --num_updates {num_updates} \
            --cache_dir {cache_dir} '''
    else:
        command = f'''python -m memory.torch_dp \
            --mode {mode} \
            --batch_size {micro_batch_size} \
            --gradient_accumulation_steps {gradient_accumulation_steps} \
            --seq_len {seq_len} \
            --model_name_or_path {model_name_or_path} \
            --num_updates {num_updates} \
            --cache_dir {cache_dir} '''
    if out_path is not None:
        command += f'--out_path {out_path}'

    print('Running command: ')
    print(command)
    out = os.system(command)
    return out


def main(
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    modes=("nonprivate", "vanilla", "layer_by_layer", "ghost", "jax"),

    seq_lens=(100,),
    num_updates=2,
    gradient_accumulation_steps=2,  # JAX fails if this is 1.
    min_micro_batch_size=2,
    max_micro_batch_size=100,
    threshold=1,  # A loose threshold for binary search.
    config_dir=f"/nlp/scr/lxuechen/prefixtune/memory/time_elapse_micro_batch_size.json",
):
    config2bsz = dict()
    for seq_len in seq_lens:
        for mode in modes:
            for model_name_or_path in model_name_or_paths:
                # Check first you can fit the minimum number of examples.
                print(f"model_name_or_path: {model_name_or_path}, mode: {mode}")

                out = _run_command(
                    mode=mode,
                    micro_batch_size=min_micro_batch_size,  # MIN
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    seq_len=seq_len,
                    model_name_or_path=model_name_or_path,
                    num_updates=num_updates,
                )
                if out != 0:  # Failed -- cannot even fit a single example.
                    config2bsz[str((model_name_or_path, mode, seq_len))] = 0
                    print("Can't even fit the min batch size, skipping...")
                    continue
                del out

                # Run binary search to get maximum batch size.

                # Don't overwrite max_micro_batch_size, min_micro_batch_size.
                this_max_micro_batch_size = max_micro_batch_size
                this_min_micro_batch_size = min_micro_batch_size
                while (this_max_micro_batch_size - this_min_micro_batch_size) > threshold:
                    mid = (this_min_micro_batch_size + this_max_micro_batch_size) // 2

                    out = _run_command(
                        mode=mode,
                        micro_batch_size=mid,  # MID
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        seq_len=seq_len,
                        model_name_or_path=model_name_or_path,
                        num_updates=num_updates,
                    )
                    if out != 0:  # Didn't fit => reduce max.
                        this_max_micro_batch_size = mid
                        print(f"Micro batch size failed: {mid}")
                    else:  # Fit => increase min.
                        this_min_micro_batch_size = mid
                        print(f"Micro batch size succeeded: {mid}")
                    del out

                # Take the left end as the maximum batch size.
                config2bsz[str((model_name_or_path, mode, seq_len))] = this_min_micro_batch_size

    utils.jdump(config2bsz, config_dir)


if __name__ == '__main__':
    fire.Fire(main)
