"""Plot the number of examples processed per unit time.

python -m memory.plot_examples_per_unit_time
"""

import collections
import os

import fire
import numpy as np

from lxuechen_utils import utils


def main(
    base_dir="/Users/xuechenli/Desktop/dump/memory/time_elapse_large_batch",
    modes=("vanilla", "layer_by_layer", "ghost", "jax"),  # modes we want to compare against nonprivate
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
):
    x = np.array(list(range(len(model_name_or_paths))))

    nonprivate_throughput = collections.defaultdict(list)  # model -> []
    for mode in modes:
        for model_name_or_path in model_name_or_paths:
            this_path = os.path.join(
                base_dir, f'nonprivate-{mode}',
                f'model_name_or_path_{model_name_or_path}_mode_nonprivate.json'
            )
            usage = utils.jload(this_path)
            throughput = (usage["batch_size"] * usage["num_updates"]) / usage["time_elapse"]
            nonprivate_throughput[model_name_or_path].append(throughput)
    # Roughly average throughputs.
    for key, item in nonprivate_throughput.items():
        nonprivate_throughput[key] = np.mean(item)

    grouped = collections.defaultdict(list)  # mode->[times for models]
    for mode in modes:
        for model_name_or_path in model_name_or_paths:
            # Get non-private time.
            this_path = os.path.join(
                base_dir, f'nonprivate-{mode}',
                f'model_name_or_path_{model_name_or_path}_mode_nonprivate.json'
            )
            usage = utils.jload(this_path)
            nonprivate_time = usage["time_elapse"]

            # Get mode time.
            this_path = os.path.join(
                base_dir, f'nonprivate-{mode}',
                f'model_name_or_path_{model_name_or_path}_mode_{mode}.json'
            )
            if not os.path.exists(this_path):
                grouped[mode].append(0)
            else:
                usage = utils.jload(this_path)
                mode_time = usage["time_elapse"]
                throughput = (nonprivate_time * nonprivate_throughput[model_name_or_path]) / mode_time
                grouped[mode].append(throughput)
    print(grouped)


if __name__ == "__main__":
    fire.Fire(main)
