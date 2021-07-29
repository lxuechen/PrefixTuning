import os

import fire

from lxuechen_utils import utils


def main(
    batch_sizes=(30, 27, 24, 21, 18, 15, 12, 9, 6, 3),
    seq_len=100,
    model_name_or_path="gpt2",
    num_updates=100,

    modes=("vanilla", "layer_by_layer", "ghost", "jax"),
    out_path=None,
):
    max_batch_sizes = dict()
    for mode in modes:
        for batch_size in batch_sizes:
            if mode == "jax":
                out = os.system(
                    f"python -m memory.jax_dp --batch_size {batch_size} --seq_len {seq_len} "
                    f"--num_updates {num_updates} --model_name_or_path {model_name_or_path}"
                )
            else:
                out = os.system(
                    f"python -m memory.torch_dp --mode {mode} --batch_size {batch_size} --seq_len {seq_len} "
                    f"--num_updates {num_updates} --model_name_or_path {model_name_or_path}"
                )
            if out == 0:  # Success
                max_batch_sizes[mode] = batch_size
                print(f'mode: {mode}, successful at batch_size: {batch_size}')

    if out_path is not None:
        utils.jdump(
            max_batch_sizes,
            out_path
        )


if __name__ == "__main__":
    # python -m memory.batch_size
    fire.Fire(main)
