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
        "layer_by_layer": "accumulate by layer",
        "ghost": "ghost",
        "jax": "JAX (JIT + VMAP)",
    }[mode]


def main(
    time_dir="/Users/xuechenli/Desktop/dump/prefixtune/memory/time_elapse",
    mem_path="/Users/xuechenli/Desktop/dump/prefixtune/memory/batch_size_exps_seq_len_00000100.json",
    model_name_or_paths=("gpt2", "gpt2-medium", "gpt2-large"),
    modes=("nonprivate", "vanilla", "layer_by_layer", "ghost", "jax"),

    width=0.14,
    img_dir="/Users/xuechenli/remote/PrefixTuning/memory/plots",
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
            plt.bar(
                x + (-2 + i) * width, this_grouped, width, label=label,
            )

        xtick_labels = ["gpt2-small", "gpt2-medium", "gpt2-large"]
        plt.xticks(x, xtick_labels)
        plt.ylabel('max batch size')
        plt.xlabel('model')
        # plt.legend()  # Don't use legend for the left plot.
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
            plt.bar(
                x + (-2 + i) * width, this_grouped, width, label=label,
            )

        xtick_labels = ["gpt2-small", "gpt2-medium", "gpt2-large"]
        plt.xticks(x, xtick_labels)
        plt.ylabel('gradient updates per minute')
        plt.xlabel('model')
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()


if __name__ == "__main__":
    # python -m memory.plot
    fire.Fire(main)
