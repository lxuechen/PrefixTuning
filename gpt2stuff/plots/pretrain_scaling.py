"""Plot the scaling behavior of fine-tuning with custom pretrained models.

python -m gpt2stuff.plots.pretrain_scaling
"""

import os

import fire

from lxuechen_utils import utils

tuning_mode_to_label = {
    'fulltune': "full",
    'scratchtune': 'scratch'
}

# untied models.
model_card_to_num_params = {
    'openai-gpt': 147621888,
    'distilgpt2': 120509952,
    'gpt2': 163037184,
    'gpt2-medium': 406286336,
    'gpt2-large': 838359040
}


def main(
    private_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/private",
    nonprivate_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/date_080421",
    n_layers=tuple(range(2, 32, 2)),

    img_dir="/Users/xuechenli/remote/PrefixTuning/gpt2stuff/plots/pretrain_scaling",
    tuning_modes=("fulltune", "scratchtune"),
    metrics=("BLEU", "tok_logprobs",),
    aspect_ratio=32,

    # Small epsilon.
    # It seems this doesn't matter so far as n_layer reaches 16.
    # private_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/date_080921",
    # n_layers=tuple(range(2, 18, 2)),
):
    os.makedirs(img_dir, exist_ok=True)

    # 080321 contains private.
    for metric in metrics:
        plots = []

        for tag, base_dir in zip(('private', 'non-private'), (private_dir, nonprivate_dir)):
            if tag == "non-private":
                continue
            for tuning_mode in tuning_modes:
                y = []
                x = []
                for n_layer in n_layers:
                    d_model = aspect_ratio * n_layer
                    folder = os.path.join(base_dir, tuning_mode, f'distilgpt2-{n_layer}-{d_model}')
                    x.append(n_layer)

                    if metric == "BLEU":
                        record_path = os.path.join(folder, 'generations_score', 'results.json')
                        record = utils.jload(record_path)
                        score = record["score"][-1][metric]
                        y.append(score)
                    elif metric == "tok_logprobs":
                        if tag != "non-private":
                            record_path = os.path.join(folder, 'log_history.json')
                            record = utils.jload(record_path)
                            score = record[-1]["eval"]["model"][metric]
                            y.append(score)
                        else:
                            record_path = os.path.join(folder, 'final_results.json')
                            record = utils.jload(record_path)
                            score = record["eval"]["model"][metric]
                            y.append(score)

                linestyle = '--' if tag == "non-private" else '-.'
                label = tuning_mode_to_label[tuning_mode] + f' ({tag})'
                plots.append({'x': n_layers, 'y': y, 'label': label, 'linestyle': linestyle})

        ylabel = "BLEU" if metric == "BLEU" else "per-token NLL"
        for img_path in (
            os.path.join(img_dir, f'metric-{metric}.pdf'),
            os.path.join(img_dir, f'metric-{metric}.png'),
        ):
            utils.plot(img_path=img_path, plots=plots,
                       options={'xlabel': f'$n_{{ \mathrm{{layer}} }}$', 'ylabel': ylabel})

    # Plot param count.
    num_params = [
        6598528, 13790208, 22164864, 32312320, 44822400, 60284928, 79289728, 102426624, 130285440, 163456000, 202528128,
        248091648, 300736384, 361052160, 429628800
    ]
    for img_path in (
        os.path.join(img_dir, f'param-count.pdf'),
        os.path.join(img_dir, f'param-count.png'),
    ):
        num_params_in_millions = [np / 1e6 for np in num_params]
        utils.plot(
            img_path=img_path,
            hlines=tuple(
                {'y': model_card_to_num_params[model_card] / 1e6,
                 'xmin': n_layers[0], 'xmax': n_layers[-1],
                 'label': model_card, 'colors': f'C{i}'}
                for i, model_card in enumerate(('gpt2', 'distilgpt2', 'gpt2-medium'), 1)
            ),
            plots=({'x': n_layers, 'y': num_params_in_millions[:len(n_layers)], 'marker': 'x'},),
            options={'xlabel': f'$n_{{ \mathrm{{layer}} }}$', 'ylabel': "number of parameters (millions)"}
        )


if __name__ == "__main__":
    fire.Fire(main)
