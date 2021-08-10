"""Plot the scaling behavior of fine-tuning with custom pretrained models.

python -m gpt2stuff.plots.pretrain_scaling_speed
"""

import os

import fire

from lxuechen_utils import utils

tuning_mode_to_label = {
    'fulltune': "full",
    'scratchtune': 'scratch'
}


def main(
    private_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/date_080721",
    nonprivate_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/date_080421",

    img_dir="/Users/xuechenli/remote/PrefixTuning/gpt2stuff/plots/pretrain_scaling_speed",
    tuning_modes=("fulltune", "scratchtune"),
    metrics=("BLEU", "tok_logprobs",),
    aspect_ratio=32,
    n_layers=range(2, 18, 2),
):
    os.makedirs(img_dir, exist_ok=True)

    # 080321 contains private.
    for metric in metrics:
        plots = []

        for tag, base_dir in zip(('private',), (private_dir,)):
            for tuning_mode in tuning_modes:
                for n_layer in n_layers:
                    d_model = aspect_ratio * n_layer
                    folder = os.path.join(base_dir, tuning_mode, f'distilgpt2-{n_layer}-{d_model}')

                    result_path = os.path.join(folder, 'log_history.json')
                    result = utils.jload(result_path)
                    x = [i["epoch"] for i in result]

                    if metric == "BLEU":
                        record_path = os.path.join(folder, 'generations_score', 'results.json')
                        record = utils.jload(record_path)
                        y = [i[metric] for i in record["score"]]

                    elif metric == "tok_logprobs":
                        record_path = os.path.join(folder, 'log_history.json')
                        record = utils.jload(record_path)
                        y = [i["eval"]["model"][metric] for i in record]

                    linestyle = 'solid' if tuning_mode == "fulltune" else '-.'
                    label = tuning_mode_to_label[tuning_mode] + f' ($n_{{ \mathrm{{layer}} }}${n_layer}) ({tag})'

                    if metric == "tok_logprobs":
                        yscale = 'log'
                    else:
                        yscale = 'linear'
                    plots.append({'x': x, 'y': y, 'label': label, 'linestyle': linestyle})

        ylabel = "BLEU" if metric == "BLEU" else "per-token NLL"
        for img_path in (
            os.path.join(img_dir, f'metric-{metric}.pdf'),
            os.path.join(img_dir, f'metric-{metric}.png'),
        ):
            utils.plot(img_path=img_path, plots=plots,
                       options={'yscale': yscale, 'xlabel': 'epoch', 'ylabel': ylabel})


if __name__ == "__main__":
    fire.Fire(main)
