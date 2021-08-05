"""Generate the table of results in dart; version 2.

python -m gpt2stuff.plots.gather_dart_v2
"""

import json
import logging
import os
import sys

import fire

from lxuechen_utils import utils

metric2name = {
    'METEOR': 'METEOR',
    'rouge1': 'ROUGE-1',
    'rouge2': 'ROUGE-2',
    'rougeL': 'ROUGE-L',
    'bleu': 'BLEU',
    'bertscore': 'BERTScore',
    'bleurt': 'BLEURT',
    'NIST': 'NIST',
}
tuning_mode2name = {
    "lineartune": "linear probing",
    "fulltune": "fine-tuning",
    "prefixtune": "prefix-tuning",
    "scratchtune": "retraining",
}


def _unwrap_raw_score(raw_score, metric):
    if metric in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        return raw_score["fmeasure"]
    elif metric in ("bertscore",):
        return raw_score["f1"]
    else:
        return raw_score


def _extract_private_score(record, metric, target_epsilon, tuning_mode):
    """Unwrap dictionary if necessary."""
    raw_score = record[target_epsilon][tuning_mode][metric]
    return _unwrap_raw_score(raw_score, metric)


def _extract_nonprivate_score(record, metric, tuning_mode):
    raw_score = record[tuning_mode][metric]
    return _unwrap_raw_score(raw_score, metric)


def json2tex(
    record,
    tuning_modes,
    target_epsilons=(2, 5, 8),
    metrics=('METEOR', "rouge1", "rouge2", "rougeL", 'bleu', 'bertscore', 'bleurt'),
    nonprivate_record=None,

    v2=True,
):
    # Record best numbers.
    best_numbers = dict()
    for metric in metrics:
        best_numbers_for_this_metric = dict()
        best_numbers[metric] = best_numbers_for_this_metric
        for target_epsilon in target_epsilons:
            best_score = -sys.maxsize
            for tuning_mode in tuning_modes:
                score = _extract_private_score(
                    record=record, target_epsilon=target_epsilon, tuning_mode=tuning_mode, metric=metric
                )
                if score > best_score:
                    best_score = score
            best_numbers_for_this_metric[target_epsilon] = best_score
            del best_score

    if not v2:
        # Write to table format.
        for tuning_mode in tuning_modes:
            tex = "\midrule \n"
            tex += "\multirow{4}[2]{*}{" + f"{tuning_mode2name[tuning_mode]}" + "}" + "\n"
            for metric in metrics:
                tex += f" & {metric2name[metric]}"
                for target_epsilon in target_epsilons:
                    score = _extract_private_score(
                        record=record, target_epsilon=target_epsilon, tuning_mode=tuning_mode, metric=metric
                    )
                    best_score = best_numbers[metric][target_epsilon]
                    if score == best_score:
                        tex += " & \\textbf{{ {:.3f} }}".format(score)
                    else:
                        tex += f" & {score:.3f}"

                if nonprivate_record is not None:
                    nonprivate_score = _extract_nonprivate_score(
                        record=nonprivate_record, tuning_mode=tuning_mode, metric=metric
                    )
                    tex += f" & {nonprivate_score:.3f}"  # Non-private no results yet.
                else:
                    tex += f" & "
                tex += "\\\\ \n"
            print(tex)
    else:
        # Version 2
        for tuning_mode in tuning_modes:
            tex = "\midrule \n"
            tex += "\multirow{4}[1]{*}{" + f"{tuning_mode2name[tuning_mode]}" + "}" + "\n"

            for target_epsilon in target_epsilons:
                tex += f" & $\epsilon={target_epsilon}$ "
                for metric in metrics:
                    score = _extract_private_score(
                        record=record, target_epsilon=target_epsilon, tuning_mode=tuning_mode, metric=metric
                    )
                    best_score = best_numbers[metric][target_epsilon]
                    if score == best_score:
                        tex += " & \\textbf{{ {:.3f} }}".format(score)
                    else:
                        tex += f" & {score:.3f}"
                tex += "\\\\ \n"

            tex += " & non-private "
            for metric in metrics:
                if nonprivate_record is not None:
                    nonprivate_score = _extract_nonprivate_score(
                        record=nonprivate_record, tuning_mode=tuning_mode, metric=metric
                    )
                    tex += f" & {nonprivate_score:.3f}"  # Non-private no results yet.
                else:
                    tex += f" & "
            tex += "\\\\ \n"

            print(tex)


# TODO: Multiple seeds.
def _main(
    base_dir="/Users/xuechenli/Desktop/dump/prefixtune/date_0720",
    nonprivate_base_dir="/Users/xuechenli/Desktop/dump/prefixtune/date_0720",

    seeds=(0,),
    model_name_or_path="gpt2",
    tuning_modes=("fulltune", "scratchtune", "prefixtune", "lineartune"),
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
            for lr in (5e-4,):
                for epochs in (50,):
                    for train_batch_size in (512,):
                        for seed in seeds:
                            epochs_str = utils.int2str(epochs)
                            lr_str = utils.float2str(lr)
                            target_epsilon_str = utils.int2str(target_epsilon)
                            train_batch_size_str = utils.int2str(train_batch_size)

                            this_folder = os.path.join(
                                base_dir,
                                f"model_name_{model_name_or_path}_nonprivate_no_tuning_mode_{tuning_mode}_"
                                f"per_example_max_grad_norm_0_10000000_"
                                f"noise_multiplier_-1_00000000_"
                                f"learning_rate_{lr_str}_"
                                f"train_batch_size_{train_batch_size_str}_"
                                f"mid_dim_00000512_"
                                f"preseqlen_00000010_"
                                f"epochs_{epochs_str}_"
                                f"target_epsilon_{target_epsilon_str}",
                                f"{seed}"
                            )
                            record_path = os.path.join(this_folder, "generations_score", "results.json")
                            gem_record_path = os.path.join(this_folder, "generations_gem_score", "results.json")
                            if not os.path.exists(record_path) or not os.path.exists(gem_record_path):
                                logging.warning(f"lost folder: {this_folder}")
                                logging.warning(
                                    f'lr={lr_str}, '
                                    f'epochs={epochs_str}, '
                                    f'target_epsilon={target_epsilon_str}, '
                                    f'tuning_mode={tuning_mode}'
                                )
                                continue

                            record = utils.jload(record_path)
                            this_score = record["score"][-1]

                            gem_record = utils.jload(gem_record_path)
                            this_gem_score = gem_record["score"][-1]
                            this_bleu = this_gem_score["bleu"]

                            # Taking the max yields same results.
                            # this_score = record["score"][-1]
                            # this_score_all = record["score"]
                            # bleu_scores = [si["BLEU"] for si in this_score_all]
                            # this_bleu = max(bleu_scores)

                            if this_bleu > best_bleu:
                                best_bleu = this_bleu
                                best_scores = {**this_score, **this_gem_score}

            results_for_this_tuning_mode.update(best_scores)

    nonprivate_record = dict()
    for seed in seeds:
        for tuning_mode in tuning_modes:
            this_folder = os.path.join(
                nonprivate_base_dir,
                f"model_name_{model_name_or_path}_"
                f"nonprivate_yes_"
                f"tuning_mode_{tuning_mode}_"
                f"learning_rate_0_00005000_"
                f"train_batch_size_00000005_"
                f"mid_dim_00000512_"
                f"preseqlen_00000010_"
                f"epochs_00000005_"
                f"target_epsilon_-0000001",
                f"{seed}",
            )
            record_path = os.path.join(this_folder, "generations_score", "results.json")
            gem_record_path = os.path.join(this_folder, "generations_gem_score", "results.json")

            if not os.path.exists(record_path) or not os.path.exists(gem_record_path):
                logging.warning(f"lost folder: {this_folder}")
                continue

            record = utils.jload(record_path)
            this_score = record["score"][-1]

            gem_record = utils.jload(gem_record_path)
            this_gem_score = gem_record["score"][-1]

            nonprivate_record[tuning_mode] = {**this_score, **this_gem_score}

    print(json.dumps(results, indent=4))
    print('---')
    json2tex(results, tuning_modes=tuning_modes, target_epsilons=target_epsilons,
             nonprivate_record=nonprivate_record)


def main(
    task="_main",
    **kwargs,
):
    if task == "_main":
        _main(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
