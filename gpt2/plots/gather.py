# python -m gpt2.plots.gather
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
):
    for tuning_mode in tuning_modes:
        tex = ""
        for metric in metrics:
            tex += f" & {metric}"
            for target_epsilon in target_epsilons:
                score = record[target_epsilon][tuning_mode][metric]
                tex += f" & {score}"
            tex += " & "  # Non-private no results yet.
            tex += "\n"
        print(tuning_mode)
        print(tex)


# TODO: EMA vs non-EMA.
# TODO: Multiple seeds.
def main(
    base_dir="/Users/xuechenli/Desktop/dump/prefixtune/date_0626",
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
                        if this_bleu > best_bleu:
                            best_bleu = this_bleu
                            best_scores = this_score
            results_for_this_tuning_mode.update(best_scores)

    print(json.dumps(results, indent=4))
    json2tex(results, tuning_modes=tuning_modes, target_epsilons=target_epsilons, )


if __name__ == "__main__":
    fire.Fire(main)
