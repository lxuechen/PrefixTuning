"""Plot ppl for various batch sizes."""

import os

import fire

from lxuechen_utils import utils

metric2label = {
    "tok_logprobs": "token NLL",
    "lin_logprobs": "line NLL",
}


def _plot(
    metric,
    base_dir="/Users/xuechenli/Desktop/dump/date_0706",
    seeds=(0,),
):
    train_plots = []
    test_plots = []

    base_lr = 1e-3
    for seed in seeds:
        for model_name_or_path in ("distilgpt2",):
            for i, train_batch_size in enumerate((1024, 512, 256, 64, 32, 16)):
                # Large epochs and large epsilon.
                for target_epsilon in (3,):
                    for epochs in (50,):
                        for tuning_mode in ("fulltune",):
                            lr = base_lr / (2 ** i)

                            epochs_str = utils.int2str(epochs)
                            lr_str = utils.float2str(lr)
                            target_epsilon_str = utils.int2str(target_epsilon)
                            train_batch_size_str = utils.int2str(train_batch_size)

                            # @formatter:off
                            train_dir = os.path.join(
                                base_dir,
                                f"model_name_{model_name_or_path}_nonprivate_no_"
                                f"tuning_mode_{tuning_mode}_"
                                f"per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_"
                                f"learning_rate_{lr_str}_"
                                f"train_batch_size_{train_batch_size_str}_"
                                f"mid_dim_00000512_preseqlen_00000010_"
                                f"epochs_{epochs_str}_target_epsilon_{target_epsilon_str}",
                                f"{seed}"
                            )
                            # @formatter:on
                            record_path = os.path.join(
                                train_dir,
                                "log_history.json"
                            )
                            record = utils.jload(record_path)

                            x = [ri["step"] for ri in record]
                            y = [ri["train"]["model"][metric] for ri in record]
                            train_plots.append(
                                {"x": x, "y": y, 'label': f"batch size={train_batch_size}"},
                            )

                            x = [ri["step"] for ri in record]
                            y = [ri["eval"]["model"][metric] for ri in record]
                            test_plots.append(
                                {"x": x, "y": y, 'label': f"batch size={train_batch_size}", 'linestyle': '-.'},
                            )

    ylabel = metric2label[metric]
    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'small_bsz', f'{metric}_train.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'small_bsz', f'{metric}_train.pdf'),
    ):
        utils.plot(
            img_path=img_path, plots=train_plots, options={'xlabel': 'Iterations', "ylabel": f'Train {ylabel}'}
        )

    for img_path in (
        os.path.join('.', 'gpt2stuff', 'plots', 'small_bsz', f'{metric}_test.png'),
        os.path.join('.', 'gpt2stuff', 'plots', 'small_bsz', f'{metric}_test.pdf'),
    ):
        utils.plot(
            img_path=img_path, plots=test_plots, options={'xlabel': 'Iterations', "ylabel": f'Test {ylabel}'}
        )


def main(
    metrics=('tok_logprobs', 'lin_logprobs'),
):
    for metric in metrics:
        _plot(metric=metric)


if __name__ == "__main__":
    # python -m gpt2stuff.plots.ppl_0706
    fire.Fire(main)
