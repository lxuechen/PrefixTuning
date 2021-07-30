"""Check batch size scaling with learning rate scaling.

Fix low and high learning rate.
"""

import os

import fire

from lxuechen_utils import utils


def main(
    seeds=(0, 1,),
    alpha=.4,
):
    for tag, lr in utils.zip_(("high lr", "mid lr", "mid low lr", "low lr"), (1e-3, 5e-4, 1e-5, 5e-6)):
        bleu_plots = []
        nll_plots = []
        nll_fbs = []
        bleu_fbs = []

        for model_name_or_path in ("distilgpt2",):
            for i, train_batch_size in enumerate((1024, 512, 256, 64, 32, 16)):
                nlls = []
                bleus = []

                for seed in seeds:
                    lr_str = utils.float2str(lr)
                    train_batch_size_str = utils.int2str(train_batch_size)

                    train_dir = os.path.join(
                        "/Users/xuechenli/Desktop/dump/prefixtune/date_0713",
                        f"model_name_"
                        f"{model_name_or_path}_nonprivate_no_tuning_mode_fulltune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_{lr_str}_train_batch_size_{train_batch_size_str}_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000003",
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

                # TODO: Hacky patch the missing one.
                for nll in nlls:
                    if len(nll) < 10:
                        nll.extend(nlls[-1][len(nll):])
                    print(len(nll))

                for bleu in bleus:
                    if len(bleu) < 10:
                        bleu.extend(bleus[-1][len(bleu):])
                    print(len(bleu))

                nll_mean, nll_std = utils.average_over_seed(nlls)
                bleu_mean, bleu_std = utils.average_over_seed(bleus)

                nll_plots.append(
                    {'x': x, 'y': nll_mean, 'label': f"batch size={train_batch_size}"}
                )
                nll_fbs.append(
                    {'x': x, 'y1': nll_mean - nll_std, 'y2': nll_mean + nll_std, 'alpha': alpha}
                )

                bleu_plots.append(
                    {'x': x, 'y': bleu_mean, 'label': f"batch size={train_batch_size}"}
                )
                bleu_fbs.append(
                    {'x': x, 'y1': bleu_mean - bleu_std, 'y2': bleu_mean + bleu_std, 'alpha': alpha}
                )

        for img_path in (
            os.path.join('.', 'gpt2stuff', 'plots', 'bsz', f'bsz_lr_joint_scaling_{tag}_nll.png'),
            os.path.join('.', 'gpt2stuff', 'plots', 'bsz', f'bsz_lr_joint_scaling_{tag}_nll.pdf'),
        ):
            kwargs = {}
            if tag != "low lr":
                kwargs["disable_legend"] = True
            legend_options = dict(prop={'size': 15})
            kwargs["legend_options"] = legend_options

            utils.plot(
                img_path=img_path,
                plots=nll_plots,
                fill_betweens=nll_fbs,
                options={'xlabel': {'xlabel': 'epoch', 'fontsize': 30},
                         'ylabel': {'ylabel': 'per-token NLL', 'fontsize': 30},
                         'ylim': (0.4, 2.5), 'yscale': 'linear'},
                **kwargs,
            )

        for img_path in (
            os.path.join('.', 'gpt2stuff', 'plots', 'bsz', f'bsz_lr_joint_scaling_{tag}_bleu.png'),
            os.path.join('.', 'gpt2stuff', 'plots', 'bsz', f'bsz_lr_joint_scaling_{tag}_bleu.pdf'),
        ):
            kwargs = {}
            if tag != "low lr":
                kwargs["disable_legend"] = True
            legend_options = dict(prop={'size': 10})
            kwargs["legend_options"] = legend_options

            utils.plot(
                img_path=img_path,
                plots=bleu_plots,
                fill_betweens=bleu_fbs,
                options={'xlabel': {'xlabel': 'epoch', 'fontsize': 20},
                         'ylabel': {'ylabel': 'BLEU', 'fontsize': 20},}
            )


if __name__ == "__main__":
    # python -m gpt2stuff.plots.bsz_fixed_lr
    fire.Fire(main)
