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


def main(
    private_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/private",
    nonprivate_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/date_080421",

    img_dir="/Users/xuechenli/remote/PrefixTuning/gpt2stuff/plots/pretrain_scaling",
    tuning_modes=("fulltune", "scratchtune"),
    metrics=("BLEU", "tok_logprobs",),
    aspect_ratio=32,
    n_layers=range(2, 32, 2),
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


if __name__ == "__main__":
    fire.Fire(main)
