import os

import fire


def main(
    num_updates=100,

    seq_lens=(100, 100, 100),
    batch_sizes=(5, 5, 5),
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    modes=("nonprivate", "vanilla", "layer_by_layer", "ghost", "jax"),

    out_dir="/nlp/scr/lxuechen/prefixtune/memory/time_elapse",
):
    os.makedirs(out_dir, exist_ok=True)

    for seq_len, batch_size, model_name_or_path in zip(seq_lens, batch_sizes, model_name_or_paths):
        for mode in modes:
            if model_name_or_path == "gpt2-large" and mode == "jax":
                continue

            print(f"model_name_or_path: {model_name_or_path}, mode: {mode}")

            out_path = os.path.join(
                out_dir, f"model_name_or_path_{model_name_or_path}_mode_{mode}.json"
            )
            if mode == "jax":
                os.system(
                    f"python -m memory.jax_dp --batch_size {batch_size} --seq_len {seq_len} "
                    f"--num_updates {num_updates} --model_name_or_path {model_name_or_path} --out_path {out_path}"
                )
            else:
                os.system(
                    f"python -m memory.torch_dp --mode {mode} --batch_size {batch_size} --seq_len {seq_len} "
                    f"--num_updates {num_updates} --model_name_or_path {model_name_or_path} --out_path {out_path}"
                )


if __name__ == '__main__':
    # python -m memory.time_elapse
    fire.Fire(main)
