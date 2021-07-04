"""Plot the scaling behavior for private-finetuning."""

import fire
import os

from lxuechen_utils import utils


def main(
    base_dir="/Users/xuechenli/Desktop/dump/date_0702_v2",
    model_name_or_paths=("distilgpt2", "gpt2", "gpt2-medium"),
    seeds=(0,),
    metric="BLEU",
    target_epsilon=3,
):
    target_epsilon_str = utils.int2str(target_epsilon)

    plots = []
    for model_name_or_path in model_name_or_paths:
        for seed in seeds:
            record_path = os.path.join(
                base_dir,
                f"model_name_{model_name_or_path}_"
                f"nonprivate_no_"
                f"tuning_mode_fulltune_"
                f"per_example_max_grad_norm_0_10000000_"
                f"noise_multiplier_-1_00000000_"
                f"learning_rate_0_00050000_"
                f"train_batch_size_00000400_"
                f"mid_dim_00000512_"
                f"preseqlen_00000010_"
                f"epochs_00000060_"
                f"target_epsilon_{target_epsilon_str}",
                f'{seed}',
                'generations_score',
                'results.json'
            )
            record = utils.jload(record_path)
            x = record['global_step']
            y = [item[metric] for item in record['score']]
            plots.append({
                'x': x, 'y': y, 'label': model_name_or_path,
            })

    img_path = os.path.join(base_dir, f'{metric}.png')
    utils.plot(
        img_path=img_path,
        plots=plots,
        options={'xlabel': 'Iterations', "ylabel": metric}
    )


if __name__ == "__main__":
    # python -m gpt2stuff.plots.scaling
    fire.Fire(main)
