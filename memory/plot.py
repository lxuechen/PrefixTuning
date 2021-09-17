import collections
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lxuechen_utils import utils

sns.set_theme(style="darkgrid")


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                 '%d' % int(height),
                 ha='center', va='bottom')


def mode2label(mode):
    return {
        "nonprivate": "non-private",
        "vanilla": "chain-rule-based (Opacus)",
        "layer_by_layer": "Lee & Kifer, 2020",
        "ghost": "ghost (ours)",
        "jax": "JAX (+jit & vmap)",
    }[mode]


def main(
    time_dir="/Users/xuechenli/Desktop/dump/prefixtune/memory/time_elapse",
    mem_path="/Users/xuechenli/Desktop/dump/prefixtune/memory/batch_size_exps_seq_len_00000100.json",
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    modes=("nonprivate", "vanilla", "layer_by_layer", "ghost", "jax"),

    width=0.14,
    img_dir="/Users/xuechenli/remote/PrefixTuning/memory/plots",
    disable_legend_on_first=True,

    legend_fontsize=18,
    label_fontsize=16,
):
    os.makedirs(img_dir, exist_ok=True)

    # Memory.
    grouped = collections.defaultdict(list)
    mem = utils.jload(mem_path)
    for mode in modes:
        for model_name_or_path in model_name_or_paths:
            usage = mem[model_name_or_path].get(mode, 0)
            grouped[mode].append(usage)

    x = np.array(list(range(len(model_name_or_paths))))

    for img_path in (
        os.path.join(img_dir, 'mem.png'),
        os.path.join(img_dir, 'mem.pdf'),
    ):
        plt.figure(dpi=300)

        for i, (mode, this_grouped) in enumerate(grouped.items()):
            label = mode2label(mode)

            xlocs = x + (-2 + i) * width
            ylocs = this_grouped
            plt.bar(xlocs, ylocs, width, label=label)

            for this_x, this_y in zip(xlocs, ylocs):
                plt.text(this_x - width / 4, this_y + 0.5, f"{int(this_y)}", fontdict=dict(fontsize=8))

        xtick_labels = ["GPT-2(-small)", "GPT-2-medium", "GPT-2-large"]
        plt.xticks(x, xtick_labels, fontsize=label_fontsize)
        plt.ylabel('maximum batch size (single TITAN RTX)', fontsize=label_fontsize)
        if not disable_legend_on_first:
            plt.legend(fontsize=legend_fontsize)  # Don't use legend for the left plot.
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    # Time.
    grouped = collections.defaultdict(list)
    for mode in modes:
        for model_name_or_path in model_name_or_paths:
            this_path = os.path.join(
                time_dir, f'model_name_or_path_{model_name_or_path}_mode_{mode}.json'
            )
            if not os.path.exists(this_path):
                grouped[mode].append(0)
            else:
                usage = utils.jload(this_path)
                # Updates per min.
                usage = usage['num_updates'] / (usage['time_elapse'] / 60)
                grouped[mode].append(usage)

    for img_path in (
        os.path.join(img_dir, 'time.png'),
        os.path.join(img_dir, 'time.pdf'),
    ):
        plt.figure(dpi=300)

        for i, (mode, this_grouped) in enumerate(grouped.items()):
            label = mode2label(mode)

            xlocs = x + (-2 + i) * width
            ylocs = this_grouped
            plt.bar(xlocs, ylocs, width, label=label)

            for this_x, this_y in zip(xlocs, ylocs):
                plt.text(this_x - width / 3, this_y + 2, f"{int(this_y)}", fontdict=dict(fontsize=8))

        xtick_labels = ["GPT-2-small", "GPT-2-medium", "GPT-2-large"]
        plt.xticks(x, xtick_labels, fontsize=label_fontsize)
        plt.ylabel('steps per minute', fontsize=label_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()


if __name__ == "__main__":
    # python -m memory.plot
    # python -m memory.plot --disable_legend_on_first False
    fire.Fire(main)
