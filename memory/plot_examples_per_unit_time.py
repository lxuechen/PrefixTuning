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
                mode_div_nonprivate = mode_time / nonprivate_time
                grouped[mode].append(mode_div_nonprivate)

    print(grouped)


if __name__ == "__main__":
    fire.Fire(main)
