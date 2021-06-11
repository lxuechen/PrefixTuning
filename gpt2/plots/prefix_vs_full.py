import os

import fire

from lxuechen_utils import utils
from lxuechen_utils import wrapper

base_dir = "/Users/xuechenli/Desktop/dump/prefixtune"


# max_grad_norms = (1, 3,)
# noise_multipliers = (0.1, )

def main(
    metric="tok_logprob",
    xaxis="steps",
):
    max_grad_norms = (1,)
    # noise_multipliers = (0.1, 0.75,)
    noise_multipliers = (0.75,)
    seed = 0
    nonprivate = "no"
    learning_rate = 1e-5

    plots = []
    for tuning_mode in ("fulltune", "prefixtune"):
        for max_grad_norm in max_grad_norms:
            for noise_multiplier in noise_multipliers:
                learning_rate_str = wrapper.float2str(learning_rate)
                per_example_max_grad_norm_str = wrapper.float2str(max_grad_norm)
                noise_multiplier_str = wrapper.float2str(noise_multiplier)

                train_dir = (
                    f"{base_dir}"
                    f"/nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_per_example_max_grad_norm_"
                    f"{per_example_max_grad_norm_str}_noise_multiplier_{noise_multiplier_str}_learning_rate_"
                    f"{learning_rate_str}"
                    f"/{seed}"
                )
                file_path = os.path.join(train_dir, 'log_history.json')
                record = utils.jload(file_path)

                x = []
                y = []
                for this_dict in record:
                    x.append(this_dict[xaxis])
                    y.append(this_dict[metric])

                linestyle = {
                    'prefixtune': '-',
                    'fulltune': '-.'
                }[tuning_mode]
                plots.append(
                    {'x': x, 'y': y,
                     'label': f"{tuning_mode}, $C=${max_grad_norm}, $\sigma=${noise_multiplier}",
                     'linestyle': linestyle},
                )

    if xaxis != "eps_rdp":
        nonprivate = "yes"
        for tuning_mode in ("fulltune", "prefixtune"):
            learning_rate_str = wrapper.float2str(learning_rate)
            train_dir = (
                f"{base_dir}"
                f"/nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_learning_rate_{learning_rate_str}"
                f"/{seed}"
            )
            file_path = os.path.join(train_dir, 'log_history.json')
            record = utils.jload(file_path)

            x = []
            y = []
            for this_dict in record:
                x.append(this_dict[xaxis])
                y.append(this_dict[metric])
            linestyle = {
                'prefixtune': '-',
                'fulltune': '-.'
            }[tuning_mode]
            plots.append(
                {'x': x, 'y': y, 'label': f"{tuning_mode}, non-private", 'linestyle': linestyle},
            )

    ylabel = {
        'tok_logprob': "token cross entropy",
        'lin_logprob': "line cross entropy",
    }[metric]
    xlabel = {
        'steps': 'Steps',
        'eps_rdp': '$\epsilon$'
    }[xaxis]
    img_path = os.path.join('.', 'gpt2', 'plots', 'prefix_vs_full', f'{metric}.png')
    utils.plot(
        img_path=img_path,
        plots=plots,
        options={'ylabel': f"Test set {ylabel}", 'yscale': 'log', 'xlabel': xlabel, 'xscale': 'log'},
    )


if __name__ == "__main__":
    # python -m gpt2.plots.prefix_vs_full --metric "tok_logprob"
    # python -m gpt2.plots.prefix_vs_full --metric "tok_logprob" --xaxis "eps_rdp"
    # python -m gpt2.plots.prefix_vs_full --metric "lin_logprob"
    fire.Fire(main)
