"""Show behavior of small batch sizes."""

import os

import fire

from lxuechen_utils import utils


def main(
    base_dir="/Users/xuechenli/Desktop/dump/date_0629",
    target_epsilon=8,
    seed=0,
    metric="BLEU",
    tuning_mode="fulltune",
    lr=5e-4,
):
    plots = []

    for train_batch_size in (5, 25, 50, 100):
        train_batch_size_str = utils.int2str(train_batch_size)
        target_epsilon_str = utils.int2str(target_epsilon)
        lr_str= utils.float2str(lr)

        record_path = os.path.join(
            base_dir,
            f"model_name_distilgpt2_nonprivate_no_tuning_mode_{tuning_mode}_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_{lr_str}_train_batch_size_{train_batch_size_str}_mid_dim_00000512_preseqlen_00000010_epochs_00000060_target_epsilon_{target_epsilon_str}",
            f"{seed}",
            "generations_score",
            "results.json"
        )
        record = utils.jload(record_path)
        x = record['global_step']
        y = [this[metric] for this in record['score']]
        plots.append(
            {'x': x, 'y': y, 'label': f'batch size={train_batch_size}'}
        )

        argparse_path = os.path.join(
            base_dir,
            f"model_name_distilgpt2_nonprivate_no_tuning_mode_{tuning_mode}_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_{lr_str}_train_batch_size_{train_batch_size_str}_mid_dim_00000512_preseqlen_00000010_epochs_00000060_target_epsilon_{target_epsilon_str}",
            f"{seed}",
            "argparse.json",
        )
        argparse = utils.jload(argparse_path)
        print(argparse["tuning_mode"])
        print(argparse["learning_rate"])
        print(argparse["model_name_or_path"])

    img_path = os.path.join('.', 'gpt2stuff', 'plots', 'small_bsz', f'{tuning_mode}')
    utils.plot(
        img_path=img_path,
        plots=plots,
        options={'ylabel': metric, 'xlabel': 'iteration',}
    )


if __name__ == '__main__':
    # python -m gpt2stuff.plots.small_bsz
    fire.Fire(main)
