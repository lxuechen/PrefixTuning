"""Identify why prefix-tuning fails.

Loop over noise. Plot with the best learning rate.
"""

import os
import sys

import fire
import seaborn as sns

from lxuechen_utils import utils


def tuning_mode_to_label(tm):
    return {
        'prefixtune': 'Prefix',
        'fulltune': "Full",
    }[tm]


def main(
    base_dir="/Users/xuechenli/Desktop/dump/prefixtune/date_0717",
    seed=0,
    metric="BLEU",
    noise_multipliers=(0.05, 0.1, 0.5, 1),  # Not including 0.01, since it's too small.
):
    # Each noise multiplier has a color; we repeat this for both prefix and full tune.
    colors = sns.color_palette()[:len(noise_multipliers)]

    # Collect plots.
    bleus = []
    nlls = []

    for tuning_mode in ("prefixtune", "fulltune"):
        for color, noise_multiplier in utils.zip_(colors, noise_multipliers):
            best_bleu = -sys.maxsize
            bleu_x = bleu_y = None

            best_nll = sys.maxsize
            nll_x = nll_y = None

            for lr in (1e-3, 5e-4, 1e-4,):
                lr_str = utils.float2str(lr)
                noise_multiplier_str = utils.float2str(noise_multiplier)
                train_dir = os.path.join(
                    base_dir,
                    f'model_name_distilgpt2_nonprivate_no_tuning_mode_'
                    f'{tuning_mode}_per_example_max_grad_norm_0_10000000_noise_multiplier_'
                    f'{noise_multiplier_str}_learning_rate_'
                    f'{lr_str}_train_batch_size_00001024_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_-0000001',
                    f'{seed}'
                )

                log_history_path = os.path.join(train_dir, 'log_history.json')
                log_history = utils.jload(log_history_path)

                score_path = os.path.join(train_dir, 'generations_score', 'results.json')
                score = utils.jload(score_path)

                x = [round(hi["epoch"]) for hi in log_history]
                this_nll = [score_i["eval"]["model"]["tok_logprobs"] for score_i in log_history]
                this_bleu = [si[metric] for si in score['score']]

                if len(x) == 8:
                    x.extend([45, 50])
                if len(x) == 9:
                    x.extend([50])

                while len(this_bleu) < 10:
                    this_bleu.append(this_bleu[-1])
                while len(this_nll) < 10:
                    this_nll.append(this_nll[-1])

                if len(x) != 10:
                    print('x fail')
                    print(train_dir)
                    continue
                if len(this_nll) != 10 or len(this_bleu) != 10:
                    print('y fail')
                    print(train_dir)
                    if len(this_nll) < 10:
                        print(this_nll)
                    if len(this_bleu) < 10:
                        print(this_bleu)
                    continue

                if this_nll[-1] < best_nll:
                    best_nll = this_nll[-1]
                    nll_x = x
                    nll_y = this_nll

                if this_bleu[-1] > best_bleu:
                    best_bleu = this_bleu[-1]
                    bleu_x = x
                    bleu_y = this_bleu

            label = f'{tuning_mode_to_label(tuning_mode)} $\sigma={noise_multiplier}$'
            linestyle = 'dotted' if tuning_mode == "prefixtune" else '-'
            bleus.append(
                {'x': bleu_x, 'y': bleu_y, 'label': label, 'linestyle': linestyle, 'color': color}
            )
            nlls.append(
                {'x': nll_x, 'y': nll_y, 'label': label, 'linestyle': linestyle, 'color': color}
            )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_bleu.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_bleu.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            plots=bleus,
            options={'xlabel': 'epoch', "ylabel": f'BLEU'},
        )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_nll.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_nll.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            plots=nlls,
            options={'xlabel': 'epoch', "ylabel": 'per-token NLL'},
        )


if __name__ == "__main__":
    # python -m gpt2stuff.plots.prefix_fails
    fire.Fire(main)
