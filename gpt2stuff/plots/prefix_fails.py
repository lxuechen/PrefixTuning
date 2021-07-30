"""Identify why prefix-tuning fails.

Loop over noise sigma. Plot with the best learning rate.

Figure in section 4.1

python -m gpt2stuff.plots.prefix_fails
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
    seeds=(0, 1),
    metric="BLEU",
    noise_multipliers=(0.05, 0.1, 0.5, 1),  # Not including 0.01, since it's too small.
    alpha=0.5,
):
    # Each noise multiplier has a color; we repeat this for both prefix and full tune.
    all_colors = sns.color_palette("Paired")[:2 * len(noise_multipliers)]
    tuning_mode_to_colors = {
        "prefixtune": [all_colors[2 * i] for i in range(len(noise_multipliers))],
        "fulltune": [all_colors[2 * i + 1] for i in range(len(noise_multipliers))],
    }

    # Collect plots.
    bleus = []
    nlls = []
    bleu_fbs = []
    nll_fbs = []

    for tuning_mode in ("prefixtune", "fulltune"):
        colors = tuning_mode_to_colors[tuning_mode]

        for color, noise_multiplier in utils.zip_(colors, noise_multipliers):

            all_bleu_x = []
            all_bleu_y = []
            all_nll_x = []
            all_nll_y = []

            for seed in seeds:

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
                        f'{lr_str}_train_batch_size_00001024_'
                        f'mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_-0000001',
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

                all_nll_x.append(nll_x)
                all_nll_y.append(nll_y)
                all_bleu_x.append(bleu_x)
                all_bleu_y.append(bleu_y)

            bleu_x, _ = utils.average_over_seed(all_bleu_x)
            bleu_y, bleu_y_std = utils.average_over_seed(all_bleu_y)

            nll_x, _ = utils.average_over_seed(all_nll_x)
            nll_y, nll_y_std = utils.average_over_seed(all_nll_y)

            label = f'{tuning_mode_to_label(tuning_mode)} $\sigma={noise_multiplier}$'
            linestyle = 'dotted' if tuning_mode == "prefixtune" else '-'

            bleus.append(
                {'x': bleu_x, 'y': bleu_y, 'label': label, 'linestyle': linestyle, 'color': color}
            )
            bleu_fbs.append(
                {'x': bleu_x, 'y1': bleu_y - bleu_y_std, 'y2': bleu_y + bleu_y_std, 'alpha': alpha, 'color': color}
            )

            nlls.append(
                {'x': nll_x, 'y': nll_y, 'label': label, 'linestyle': linestyle, 'color': color}
            )
            nll_fbs.append(
                {'x': nll_x, 'y1': nll_y - nll_y_std, 'y2': nll_y + nll_y_std, 'alpha': alpha, 'color': color}
            )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_bleu.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_bleu.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            plots=bleus,
            fill_betweens=bleu_fbs,
            options={'xlabel': 'epoch', "ylabel": f'BLEU'},
        )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_nll.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'prefix_fails_nll.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            plots=nlls,
            fill_betweens=nll_fbs,
            options={'xlabel': 'epoch', "ylabel": 'per-token NLL'},
        )


if __name__ == "__main__":
    fire.Fire(main)
