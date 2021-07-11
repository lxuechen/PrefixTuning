"""Check batch size scaling with learning rate scaling."""

import os

import fire

from lxuechen_utils import utils


def main(
    seeds=(0, 1),
    base_lr=1e-3,
):
    bleu_plots = []
    nll_plots = []
    for model_name_or_path in ("distilgpt2",):
        for i, train_batch_size in enumerate((1024, 512, 256, 64, 32, 16)):
            nlls = []
            bleus = []

            for seed in seeds:
                lr = base_lr / (2 ** i)

                lr_str = utils.float2str(lr)
                train_batch_size_str = utils.int2str(train_batch_size)

                train_dir = os.path.join(
                    "/Users/xuechenli/Desktop/dump/prefixtune/date_0708",
                    f"model_name_distilgpt2_nonprivate_no_tuning_mode_fulltune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_{lr_str}_train_batch_size_{train_batch_size_str}_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000003",
                    f"{seed}",
                )
                log_history_path = os.path.join(train_dir, 'log_history.json')
                log_history = utils.jload(log_history_path)

                x = [round(hi["epoch"]) for hi in log_history]

                nll = [score_i["eval"]["model"]["tok_logprobs"] for score_i in log_history]
                nlls.append(nll)

                record_path = os.path.join(train_dir, "generations_score", "results.json")
                record = utils.jload(record_path)

                score = record['score']
                bleu = [score_i["BLEU"] for score_i in score]
                bleus.append(bleu)

            nll_mean, nll_std = utils.average_over_seed(nlls)
            bleu_mean, bleu_std = utils.average_over_seed(bleus)

            nll_plots.append(
                {'x': x, 'y': nll_mean, 'yerr': nll_std, 'label': f"batch size={train_batch_size}"}
            )
            bleu_plots.append(
                {'x': x, 'y': bleu_mean, 'yerr': bleu_std, 'label': f"batch size={train_batch_size}"}
            )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'bsz', 'bsz_lr_joint_scaling_nll.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'bsz', 'bsz_lr_joint_scaling_nll.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            errorbars=nll_plots,
            options={'xlabel': 'epoch', 'ylabel': 'per-token NLL'}
        )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'bsz', 'bsz_lr_joint_scaling_bleu.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'bsz', 'bsz_lr_joint_scaling_bleu.pdf'),
    ):
        utils.plot(
            img_path=img_path,
            errorbars=bleu_plots,
            options={'xlabel': 'epoch', 'ylabel': 'BLEU'}
        )


if __name__ == "__main__":
    # python -m gpt2stuff.plots.bsz
    fire.Fire(main)
