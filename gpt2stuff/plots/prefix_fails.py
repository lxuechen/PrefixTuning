"""Identify why prefix-tuning fails.

Loop over noise. Plot with the best learning rate.
"""

import os
import sys

import fire

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
):
    # Collect plots.
    bleus = []
    nlls = []

    for tuning_mode in ("prefixtune", "fulltune"):
        # for noise_multiplier in (0.01, 0.05, 0.1, 0.5, 1):
        for noise_multiplier in (0.01, 0.05, 0.1, 0.5,):
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

                x = [round(hi["epoch"]) for hi in log_history]
                y = [score_i["eval"]["model"]["tok_logprobs"] for score_i in log_history]
                if y[-1] < best_nll:
                    nll_x = x
                    nll_y = y
                del y

                score_path = os.path.join(train_dir, 'generations_score', 'results.json')

                score = utils.jload(score_path)
                y = [si[metric] for si in score['score']]
                if y[-1] > best_bleu:
                    bleu_x = x
                    bleu_y = y
                del x, y

            label = f'{tuning_mode_to_label(tuning_mode)} $\sigma={noise_multiplier}$'
            linestyle = 'dotted' if tuning_mode == "prefixtune" else '-'
            bleus.append(
                {'x': bleu_x, 'y': bleu_y, 'label': label, 'linestyle': linestyle}
            )
            nlls.append(
                {'x': nll_x, 'y': nll_y, 'label': label, 'linestyle': linestyle}
            )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'bleu.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'bleu.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            plots=bleus,
            options={'xlabel': 'epoch', "ylabel": f'BLEU'},
        )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'nll.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'prefix_fails', 'nll.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            plots=nlls,
            options={'xlabel': 'epoch', "ylabel": 'per-token NLL'},
        )


if __name__ == "__main__":
    # python -m gpt2stuff.plots.prefix_fails
    fire.Fire(main)
