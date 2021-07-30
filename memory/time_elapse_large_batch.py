"""Profile what happens with large batches and gradient accumulation.

python -m memory.time_elapse_large_batch
"""
import math
import os

import fire

from lxuechen_utils import utils


def main(
    num_updates=100,

    seq_lens=(100, 100, 100),
    batch_sizes=(512, 512, 512),  # This is different from previous file.
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    modes=("nonprivate", "vanilla", "layer_by_layer", "ghost", "jax"),

    out_dir="/nlp/scr/lxuechen/prefixtune/memory/time_elapse",
    config_dir="/nlp/scr/lxuechen/prefixtune/memory/batch_size_exps_seq_len_00000100.json",
):
    config = utils.jload(config_dir)
    # Get max batch sizes; only for seq_len = 100. Use this for computing gradient accumulation steps.
    config2bsz = {
        (model_name_or_path, mode): config[model_name_or_path].get(mode, 0)
        for model_name_or_path in model_name_or_paths
        for mode in modes
    }

    os.makedirs(out_dir, exist_ok=True)

    for seq_len, batch_size, model_name_or_path in zip(seq_lens, batch_sizes, model_name_or_paths):
        for mode in modes:
            if model_name_or_path == "gpt2-large" and mode == "jax":
                continue

            print(f"model_name_or_path: {model_name_or_path}, mode: {mode}")

            micro_batch_size = config2bsz[(model_name_or_path, mode)]
            gradient_accumulation_steps = math.ceil(batch_size / micro_batch_size)

            out_path = os.path.join(
                out_dir, f"model_name_or_path_{model_name_or_path}_mode_{mode}.json"
            )

            if mode == "jax":
                os.system(
                    f"python -m memory.jax_dp "
                    f"--batch_size {micro_batch_size} "
                    f"--gradient_accumulation_steps {gradient_accumulation_steps} "
                    f"--seq_len {seq_len} "
                    f"--num_updates {num_updates} "
                    f"--model_name_or_path {model_name_or_path} "
                    f"--out_path {out_path}"
                )
            else:
                os.system(
                    f"python -m memory.torch_dp "
                    f"--mode {mode} "
                    f"--batch_size {micro_batch_size} "
                    f"--gradient_accumulation_steps {gradient_accumulation_steps} "
                    f"--seq_len {seq_len} "
                    f"--num_updates {num_updates} "
                    f"--model_name_or_path {model_name_or_path} "
                    f"--out_path {out_path}"
                )


if __name__ == '__main__':
    fire.Fire(main)
