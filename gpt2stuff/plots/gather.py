# python -m gpt2stuff.plots.gather
import json
import logging
import os
import sys

import fire

from lxuechen_utils import utils


def json2tex(
    record,
    tuning_modes,
    target_epsilons=(2, 5, 8),
    metrics=('BLEU', 'NIST', 'METEOR', 'ROUGE_L', 'CIDEr'),
    nonprivate_record=None,
):
    best_numbers = dict()
    for metric in metrics:
        best_numbers_for_this_metric = dict()
        best_numbers[metric] = best_numbers_for_this_metric
        for target_epsilon in target_epsilons:
            best_score = -sys.maxsize
            for tuning_mode in tuning_modes:
                score = record[target_epsilon][tuning_mode][metric]
                if score > best_score:
                    best_score = score
            best_numbers_for_this_metric[target_epsilon] = best_score
            del best_score

    for tuning_mode in tuning_modes:
        tex = ""
        for metric in metrics:
            tex += f" & {metric}"
            for target_epsilon in target_epsilons:
                score = record[target_epsilon][tuning_mode][metric]
                best_score = best_numbers[metric][target_epsilon]
                if score == best_score:
                    tex += " & \\textbf{{ {:.4f} }}".format(score)
                else:
                    tex += f" & {score:.4f}"

            if nonprivate_record is not None:
                nonprivate_score = nonprivate_record[tuning_mode][metric]
                tex += f" & {nonprivate_score:.4f}"  # Non-private no results yet.
            else:
                tex += f" & "
            tex += "\\\\ \n"
        print(tuning_mode)
        print(tex)


# TODO: EMA vs non-EMA.
# TODO: Multiple seeds.
def _main(
    base_dir="/Users/xuechenli/Desktop/dump/prefixtune/date_0626",
    nonprivate_base_dir="/Users/xuechenli/Desktop/dump/prefixtune/date_0702",

    seeds=(0,),

    tuning_modes=("fulltune", "scratchtune", "prefixtune"),
    target_epsilons=(2, 5, 8),
):
    results = dict()
    for target_epsilon in target_epsilons:
        results_for_this_epsilon = dict()
        results[target_epsilon] = results_for_this_epsilon

        for tuning_mode in tuning_modes:
            results_for_this_tuning_mode = dict()
            results_for_this_epsilon[tuning_mode] = results_for_this_tuning_mode

            best_bleu = -sys.maxsize
            best_scores = None
            for lr in (5e-4, 1e-4, 5e-3, 1e-3):
                for epochs in (60, 40, 20):
                    for seed in seeds:
                        epochs_str = utils.int2str(epochs)
                        lr_str = utils.float2str(lr)
                        target_epsilon_str = utils.int2str(target_epsilon)

                        record_path = os.path.join(
                            base_dir,
                            f"model_name_distilgpt2_nonprivate_no_tuning_mode_{tuning_mode}_"
                            f"per_example_max_grad_norm_0_10000000_"
                            f"noise_multiplier_-1_00000000_"
                            f"learning_rate_{lr_str}_"
                            f"train_batch_size_00000400_"
                            f"mid_dim_00000512_"
                            f"preseqlen_00000010_"
                            f"epochs_{epochs_str}_"
                            f"target_epsilon_{target_epsilon_str}"
                            f"/{seed}"
                            f"/generations_score"
                            f"/results.json"
                        )
                        if not os.path.exists(record_path):
                            logging.warning(f'Lost record_path {record_path}')
                            logging.warning(
                                f'lr={lr_str}, '
                                f'epochs={epochs_str}, '
                                f'target_epsilon={target_epsilon_str}, '
                                f'tuning_mode={tuning_mode}'
                            )
                            continue

                        record = utils.jload(record_path)
                        this_score = record["score"][-1]
                        this_bleu = this_score["BLEU"]

                        # Taking the max yields same results.
                        # this_score = record["score"][-1]
                        # this_score_all = record["score"]
                        # bleu_scores = [si["BLEU"] for si in this_score_all]
                        # this_bleu = max(bleu_scores)

                        if this_bleu > best_bleu:
                            best_bleu = this_bleu
                            best_scores = this_score
            results_for_this_tuning_mode.update(best_scores)

    nonprivate_record = dict()
    for seed in seeds:
        for tuning_mode in tuning_modes:
            record_path = os.path.join(
                nonprivate_base_dir,
                "model_name_distilgpt2_"
                "nonprivate_yes_"
                f"tuning_mode_{tuning_mode}_"
                "learning_rate_0_00005000_"
                "train_batch_size_00000005_"
                "mid_dim_00000512_"
                "preseqlen_00000010_"
                "epochs_00000005_"
                "target_epsilon_-0000001"
                f"/{seed}"
                f"/generations_score"
                f"/results.json"
            )
            if not os.path.exists(record_path):
                logging.warning(f'Lost record_path {record_path}')
                continue
            record = utils.jload(record_path)
            this_score = record["score"][-1]
            nonprivate_record[tuning_mode] = this_score

    print(json.dumps(results, indent=4))
    json2tex(results, tuning_modes=tuning_modes, target_epsilons=target_epsilons,
             nonprivate_record=nonprivate_record)


def ema(
    base_dir="/Users/xuechenli/Desktop/dump/prefixtune/date_0626",
    seeds=(0,),

    tuning_modes=("fulltune", "scratchtune", "prefixtune"),
    target_epsilons=(2, 5, 8),
    lrs=(5e-4, 1e-4, 5e-3, 1e-3),
    epochslist=(60, 40, 20),

    img_dir="/Users/xuechenli/Desktop/plots",
    metric="BLEU",
):
    """Check the evolution of BLEU scores for EMA vs non-EMA."""
    for target_epsilon in target_epsilons:
        for tuning_mode in tuning_modes:
            for lr in lrs:
                for epochs in epochslist:
                    for seed in seeds:
                        epochs_str = utils.int2str(epochs)
                        lr_str = utils.float2str(lr)
                        target_epsilon_str = utils.int2str(target_epsilon)

                        record_path = os.path.join(
                            base_dir,
                            f"model_name_distilgpt2_nonprivate_no_tuning_mode_{tuning_mode}_"
                            f"per_example_max_grad_norm_0_10000000_"
                            f"noise_multiplier_-1_00000000_"
                            f"learning_rate_{lr_str}_"
                            f"train_batch_size_00000400_"
                            f"mid_dim_00000512_"
                            f"preseqlen_00000010_"
                            f"epochs_{epochs_str}_"
                            f"target_epsilon_{target_epsilon_str}"
                            f"/{seed}"
                            f"/generations_score"
                            f"/results.json"
                        )
                        if not os.path.exists(record_path):
                            logging.warning(f'Lost record_path {record_path}')
                            logging.warning(
                                f'lr={lr_str}, '
                                f'epochs={epochs_str}, '
                                f'target_epsilon={target_epsilon_str}, '
                                f'tuning_mode={tuning_mode}'
                            )
                            continue

                        plots = []

                        record = utils.jload(record_path)
                        global_step = record["global_step"]
                        score = record["score"]
                        score = [score_i[metric] for score_i in score]
                        plots.append({'x': global_step, 'y': score, 'label': "vanilla"})

                        # EMA score.
                        record_path = os.path.join(
                            base_dir,
                            f"model_name_distilgpt2_nonprivate_no_tuning_mode_{tuning_mode}_"
                            f"per_example_max_grad_norm_0_10000000_"
                            f"noise_multiplier_-1_00000000_"
                            f"learning_rate_{lr_str}_"
                            f"train_batch_size_00000400_"
                            f"mid_dim_00000512_"
                            f"preseqlen_00000010_"
                            f"epochs_{epochs_str}_"
                            f"target_epsilon_{target_epsilon_str}"
                            f"/{seed}"
                            f"/generations_ema_score"
                            f"/results.json"
                        )
                        if not os.path.exists(record_path):
                            logging.warning(f'Lost record_path {record_path}')
                            logging.warning(
                                f'lr={lr_str}, '
                                f'epochs={epochs_str}, '
                                f'target_epsilon={target_epsilon_str}, '
                                f'tuning_mode={tuning_mode}'
                            )
                            continue

                        record = utils.jload(record_path)
                        global_step = record["global_step"]
                        ema_score = record["score"]
                        ema_score = [ema_score_i["BLEU"] for ema_score_i in ema_score]
                        plots.append({'x': global_step, 'y': ema_score, 'label': 'ema'})

                        img_path = os.path.join(
                            img_dir,
                            f"tuning_mode_{tuning_mode}_"
                            f"learning_rate_{lr_str}_"
                            f"epochs_{epochs_str}_"
                            f"target_epsilon_{target_epsilon_str}"
                        )
                        utils.plot(
                            img_path=img_path,
                            plots=plots,
                            options={'ylabel': metric, 'xlabel': "Iterations"}
                        )
                        print(score, ema_score)


def main(
    task="_main",
    **kwargs,
):
    if task == "_main":
        _main(**kwargs)
    elif task == "ema":
        ema()


if __name__ == "__main__":
    fire.Fire(main)
