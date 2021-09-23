"""Plot the number of examples processed per unit time.

python -m memory.plot_examples_per_unit_time
"""

import collections
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lxuechen_utils import utils

sns.set_theme(style="darkgrid")


def mode2label(mode):
    return {
        "nonprivate": "non-private",
        "vanilla": "chain-rule-based (Opacus)",
        "layer_by_layer": "Lee & Kifer, 2020",
        "ghost": "ghost (ours)",
        "jax": "JAX (+jit & vmap)",
    }[mode]


def main(
    base_dir="/Users/xuechenli/Desktop/dump/memory/time_elapse_large_batch",
    modes=("vanilla", "layer_by_layer", "ghost", "jax"),  # modes we want to compare against nonprivate
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    img_dir="/Users/xuechenli/remote/PrefixTuning/memory/plots",
    width=0.14,
    legend_fontsize=18,
    label_fontsize=16,
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
    grouped["nonprivate"] = [
        nonprivate_throughput[model] for model in model_name_or_paths
    ]
    print(grouped)

    for img_path in (
        os.path.join(img_dir, 'throughput.png'),
        os.path.join(img_dir, 'throughput.pdf'),
    ):
        plt.figure(dpi=300)

        for i, mode in enumerate(("nonprivate", "vanilla", "layer_by_layer", "jax", "ghost",)):
            this_grouped = grouped[mode]
            label = mode2label(mode)

            xlocs = x + (-2 + i) * width
            ylocs = this_grouped
            plt.bar(xlocs, ylocs, width, label=label)

            for this_x, this_y in zip(xlocs, ylocs):
                plt.text(this_x - width / 3, this_y + 2, f"{int(this_y)}", fontdict=dict(fontsize=8))

        xtick_labels = ["GPT-2", "GPT-2-medium", "GPT-2-large"]
        plt.xticks(x, xtick_labels, fontsize=label_fontsize)
        plt.ylabel('examples per second', fontsize=label_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()


if __name__ == "__main__":
    fire.Fire(main)
