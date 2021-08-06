"""Plot the scaling behavior of fine-tuning with custom pretrained models.

python -m gpt2stuff.plots.pretrain_scaling
"""

import os

import fire

from lxuechen_utils import utils

tuning_mode_to_label = {
    'fulltune': "Full",
    'scratchtune': 'Scratch'
}


def main(
    private_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/date_080321",
    nonprivate_dir="/Users/xuechenli/Desktop/dump_a100/prefixtune/date_080421",

    img_dir="/Users/xuechenli/remote/PrefixTuning/gpt2stuff/plots/pretrain_scaling",
    tuning_modes=("fulltune", "scratchtune"),
    metrics=("BLEU", "tok_logprobs"),
    aspect_ratio=32,
    n_layers=range(2, 16, 2),
):
    os.makedirs(img_dir, exist_ok=True)

    # 080321 contains private.
    base_dir = private_dir
    for metric in metrics:
        plots = []
        for tuning_mode in tuning_modes:
            y = []
            x = []
            for n_layer in n_layers:
                d_model = aspect_ratio * n_layer
                folder = os.path.join(base_dir, tuning_mode, f'distilgpt2-{n_layer}-{d_model}')
                x.append(n_layer)

                if metric == "BLEU":
                    record_path = os.path.join(
                        folder, 'generations_score', 'results.json'
                    )
                    record = utils.jload(record_path)
                    score = record["score"][-1][metric]
                    y.append(score)
                elif metric == "tok_logprobs":
                    record_path = os.path.join(folder, 'log_history.json')
                    record = utils.jload(record_path)
                    score = record[-1]["eval"]["model"][metric]
                    y.append(score)

            label = tuning_mode_to_label[tuning_mode]
            plots.append({'x': n_layers, 'y': y, 'label': label})

        for img_path in (
            os.path.join(img_dir, f'metric-{metric}.pdf'),
            os.path.join(img_dir, f'metric-{metric}.png'),
        ):
            utils.plot(img_path=img_path, plots=plots)


if __name__ == "__main__":
    fire.Fire(main)
