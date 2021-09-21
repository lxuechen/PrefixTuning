"""Profile what happens with large batches and gradient accumulation.

python -m memory.time_elapse_large_batch
"""
import os

import fire

from lxuechen_utils import utils


def main(

    seq_lens=(100,),

    num_updates=2,
    batch_sizes=(80,),  # This is different from previous file.

    model_name_or_paths=("gpt2-large",),
    modes=("layer_by_layer", "ghost",),

    out_dir="/nlp/scr/lxuechen/prefixtune/memory/time_elapse_large_batch",
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

    micro_batch_sizes = (8, 10)
    for seq_len, batch_size, model_name_or_path in zip(seq_lens, batch_sizes, model_name_or_paths):
        for mode, micro_batch_size in zip(modes, micro_batch_sizes):
            if micro_batch_size == 0:
                continue

            print(f"model_name_or_path: {model_name_or_path}, mode: {mode}")
            gradient_accumulation_steps = batch_size // micro_batch_size
            actual_num_updates = num_updates * gradient_accumulation_steps
            out_path = os.path.join(
                out_dir, f"model_name_or_path_{model_name_or_path}_mode_{mode}.json"
            )

            if mode == "jax":
                os.system(
                    f"python -m memory.jax_dp_grad_accumulation "
                    f"--batch_size {micro_batch_size} "
                    f"--gradient_accumulation_steps {gradient_accumulation_steps} "
                    f"--seq_len {seq_len} "
                    f"--model_name_or_path {model_name_or_path} "
                    f"--out_path {out_path} "
                    f"--num_updates {actual_num_updates}"
                )
            else:
                os.system(
                    f"python -m memory.torch_dp "
                    f"--mode {mode} "
                    f"--batch_size {micro_batch_size} "
                    f"--gradient_accumulation_steps {gradient_accumulation_steps} "
                    f"--seq_len {seq_len} "
                    f"--model_name_or_path {model_name_or_path} "
                    f"--out_path {out_path} "
                    f"--num_updates {actual_num_updates}"
                )


if __name__ == '__main__':
    fire.Fire(main)
