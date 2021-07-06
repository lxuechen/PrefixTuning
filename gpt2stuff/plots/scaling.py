"""Plot the scaling behavior for private-finetuning."""

import os

import fire
import numpy as np

from lxuechen_utils import utils

# Number of parameters for the untied models! Tied models are different.
MODEL2NPARAMS = {
    "openai-gpt": 147.6219,
    "distilgpt2": 120.5100,
    "gpt2": 163.0372,
    "gpt2-medium": 406.2863,
    "gpt2-large": 838.3590,
}


def _main(
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


def sigmoid(t):
    return np.exp(t) / (1 + np.exp(t))


def fig1(
    base_dir="/Users/xuechenli/Desktop/dump/date_0702_v2",
    nonprivate_base_dir="/Users/xuechenli/Desktop/dump/date_0702_v4",
    model_name_or_paths=("distilgpt2", "gpt2", "gpt2-medium"),
    metric="BLEU",
    target_epsilon=3,
    target_global_step=5000,
    seed=0,
):
    """First attempt at creating a figure 1."""
    target_epsilon_str = utils.int2str(target_epsilon)

    # private.
    y = []
    for model_name_or_path in model_name_or_paths:
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
        xx = record['global_step']
        yy = [item[metric] for item in record['score']]

        index = xx.index(target_global_step)
        y.append(yy[index])

    scatters = []

    x = [MODEL2NPARAMS[model_name_or_path] for model_name_or_path in model_name_or_paths]
    x, y = [np.array(l) for l in (x, y)]
    c = sigmoid((y - np.mean(y)) * 20)
    # Use `vmin` to avoid default color normalization.
    scatters.append({'x': x, 'y': y, 'c': c, 'cmap': 'Blues', 'edgecolors': 'none', 'vmin': 0.1})
    del x, y, c

    # non-private.
    y = []
    for model_name_or_path in model_name_or_paths:
        record_path = os.path.join(
            nonprivate_base_dir,
            f"model_name_"
            f"{model_name_or_path}_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001",
            f'{seed}',
            'generations_score',
            'results.json'
        )
        record = utils.jload(record_path)
        xx = record['global_step']
        yy = [item[metric] for item in record['score']]

        index = xx.index(target_global_step)
        y.append(yy[index])
    x = [MODEL2NPARAMS[model_name_or_path] for model_name_or_path in model_name_or_paths]
    x, y = [np.array(l) for l in (x, y)]
    c = sigmoid((y - np.mean(y)) * 20)
    # Use `vmin` to avoid default color normalization.
    scatters.append({'x': x, 'y': y, 'c': c, 'cmap': 'Blues', 'edgecolors': 'none', 'vmin': 0.1})
    del x, y, c

    img_path = os.path.join(base_dir, 'fig1.pdf')
    utils.plot(
        img_path=img_path,
        scatters=scatters,
        options={'ylabel': 'BLEU', 'xlabel': "number of parameters (millions)"}
    )


def main(task="fig1", **kwargs):
    if task == "fig1":
        fig1(**kwargs)
    elif task == "_main":
        _main(**kwargs)
    else:
        raise ValueError


if __name__ == "__main__":
    # python -m gpt2stuff.plots.scaling
    fire.Fire(main)
