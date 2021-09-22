"""Profile what happens with large batches and gradient accumulation.

python -m memory.time_elapse_large_batch
"""
import math
import os

import fire

from lxuechen_utils import utils


def lcm(l):
    """Compute the gcd for a list of numbers.

    Skip 0s.
    """
    acc = 1
    for i in l:
        if i == 0:
            continue
        acc = acc * i // math.gcd(acc, i)
    return acc


def main(
    seq_lens=(100,),

    num_updates=3,

    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    modes=("nonprivate", "vanilla", "layer_by_layer", "ghost", "jax"),

    out_dir="/nlp/scr/lxuechen/prefixtune/memory/time_elapse_large_batch",
    cache_dir="/nlp/scr/lxuechen/prefixtune/memory/cache",
    config_dir=f"/nlp/scr/lxuechen/prefixtune/memory/time_elapse_micro_batch_size.json",
):
    os.makedirs(out_dir, exist_ok=True)

    config2bsz = utils.jload(config_dir)

    for seq_len in seq_lens:
        for model_name_or_path in model_name_or_paths:
            # Get the lcm of all methods, and use that as the actual batch size.
            micro_batch_size_l = [
                config2bsz[str((model_name_or_path, mode, seq_len))]
                for mode in modes
            ]
            update_batch_size = lcm(micro_batch_size_l)

            for mode, micro_batch_size in utils.zip_(modes, micro_batch_size_l):
                if micro_batch_size == 0:
                    continue

                print(f"model_name_or_path: {model_name_or_path}, mode: {mode}")
                gradient_accumulation_steps = update_batch_size // micro_batch_size
                actual_num_updates = num_updates * gradient_accumulation_steps
                out_path = os.path.join(
                    out_dir, f"model_name_or_path_{model_name_or_path}_mode_{mode}.json"
                )

                if mode == "jax":
                    command = f'''python -m memory.jax_dp_grad_accumulation \
                        --batch_size {micro_batch_size} \
                        --gradient_accumulation_steps {gradient_accumulation_steps} \
                        --seq_len {seq_len} \
                        --model_name_or_path {model_name_or_path} \
                        --out_path {out_path} \
                        --num_updates {actual_num_updates} \
                        --cache_dir {cache_dir} '''
                else:
                    command = f'''python -m memory.torch_dp \
                        --mode {mode} \
                        --batch_size {micro_batch_size} \
                        --gradient_accumulation_steps {gradient_accumulation_steps} \
                        --seq_len {seq_len} \
                        --model_name_or_path {model_name_or_path} \
                        --out_path {out_path} \
                        --num_updates {actual_num_updates} \
                        --cache_dir {cache_dir} '''
                os.system(command)


if __name__ == '__main__':
    fire.Fire(main)
