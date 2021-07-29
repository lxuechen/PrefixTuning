import os

import fire

from lxuechen_utils import utils


def main(
    batch_sizes=tuple(range(50, 0, -2)),
    seq_len=100,
    model_name_or_path="gpt2",
    num_updates=3,  # You don't need to many updates, since this focuses on memory scaling.

    modes=("vanilla", "layer_by_layer", "ghost", "jax"),
    out_dir=None,
):
    max_batch_sizes = dict()
    results = dict(
        max_batch_sizes=max_batch_sizes,
        batch_sizes=batch_sizes,
        seq_len=seq_len,
        model_name_or_path=model_name_or_path,
        num_updates=num_updates,
    )
    for mode in modes:
        for batch_size in batch_sizes:
            print(f"mode: {mode}, batch_size={batch_size}")

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
                print(f'mode: {mode}, batch_size: {batch_size}')
                break

    if out_dir is not None:
        seq_len_str = utils.int2str(seq_len)
        out_path = os.path.join(out_dir, f'batch_size_exps_seq_len_{seq_len_str}.json')
        utils.jdump(results, out_path)


if __name__ == "__main__":
    # python -m memory.batch_size --out_dir "/nlp/scr/lxuechen/prefixtune/memory"
    fire.Fire(main)
