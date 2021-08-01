"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    0731
purpose:
    Sanitize DART generation files. Make them compatible with GEM-metrics.

    My adapted version fixes several bugs.
    https://github.com/lxuechen/GEM-metrics
notes:
    These are CPU-only jobs. So submit to john.
run:
    python -m gpt2stuff.launchers.sanitize_dart_073121 --mode "submit"
    python -m gpt2stuff.launchers.sanitize_dart_073121 --mode "local"
"""

import os

import fire

from lxuechen_utils import utils
from . import wrapper


def main(
    seeds=(0,),
    mode=wrapper.Mode.local,
    conda_env="lxuechen-prefix-tuning-pg",
):
    if mode == wrapper.Mode.local:
        pass
    elif mode == wrapper.Mode.submit:
        ref_path = (
            "/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-test.json"
        )

        commands = ""

        max_grad_norm = 0.1
        noise_multiplier = -1
        nonprivate = "no"

        for seed in seeds:
            for model_name_or_path in ("gpt2",):
                for train_batch_size in (512,):
                    for epochs in (50,):
                        for lr in (5e-4,):
                            for target_epsilon in (3, 5, 8, 2):
                                for tuning_mode in ("fulltune", "scratchtune", "prefixtune", "lineartune"):
                                    epochs_str = utils.int2str(epochs)
                                    lr_str = utils.float2str(lr)
                                    target_epsilon_str = utils.int2str(target_epsilon)
                                    train_batch_size_str = utils.int2str(train_batch_size)
                                    max_grad_norm_str = utils.float2str(max_grad_norm)
                                    noise_multiplier_str = wrapper.float2str(noise_multiplier)

                                    # @formatter:off
                                    train_dir = (
                                        f"/nlp/scr/lxuechen/prefixtune/date_0720"
                                        f"/model_name_{model_name_or_path}_"
                                        f"nonprivate_{nonprivate}_"
                                        f"tuning_mode_{tuning_mode}_"
                                        f"per_example_max_grad_norm_{max_grad_norm_str}_"
                                        f"noise_multiplier_{noise_multiplier_str}_"
                                        f"learning_rate_{lr_str}_"
                                        f"train_batch_size_{train_batch_size_str}_"
                                        f"mid_dim_00000512_"
                                        f"preseqlen_00000010_"
                                        f"epochs_{epochs_str}_"
                                        f"target_epsilon_{target_epsilon_str}"
                                        f"/{seed}"
                                    )
                                    # @formatter:on

                                    gen_dirs = (
                                        os.path.join(train_dir, "generations_model/eval"),
                                    )
                                    log_paths = (
                                        os.path.join(train_dir, 'log_gem_sanitize.out'),
                                    )

                                    for gen_dir, log_path in utils.zip_(gen_dirs, log_paths):
                                        command = (
                                            f"python -m gpt2stuff.eval.dart2gem "
                                            f"--ref_path {ref_path} "
                                            f"--gen_dir {gen_dir} "
                                            f"--task 'convert_gen' "
                                        )
                                        command = wrapper.cpu_job_wrapper(
                                            command,
                                            train_dir=train_dir, conda_env=conda_env, hold_job=False, log_path=log_path,
                                        )
                                        command += '\n'
                                        commands += command

        # Non-private.
        target_epsilon = -1
        nonprivate = "yes"

        for seed in seeds:
            for model_name_or_path in ("gpt2",):
                for train_batch_size in (5,):
                    for epochs in (5,):
                        for lr in (5e-5,):
                            for tuning_mode in ("fulltune", "scratchtune", "prefixtune", "lineartune",):
                                epochs_str = utils.int2str(epochs)
                                lr_str = utils.float2str(lr)
                                target_epsilon_str = utils.int2str(target_epsilon)
                                train_batch_size_str = utils.int2str(train_batch_size)
                                max_grad_norm_str = utils.float2str(max_grad_norm)
                                noise_multiplier_str = wrapper.float2str(noise_multiplier)

                                # @formatter:off
                                train_dir = (
                                    f"/nlp/scr/lxuechen/prefixtune/date_0720"
                                    f"/model_name_gpt2_"
                                    f"nonprivate_{nonprivate}_"
                                    f"tuning_mode_{tuning_mode}_"
                                    f"learning_rate_{lr_str}_"
                                    f"train_batch_size_{train_batch_size_str}_"
                                    f"mid_dim_00000512_"
                                    f"preseqlen_00000010_"
                                    f"epochs_{epochs_str}_"
                                    f"target_epsilon_{target_epsilon_str}/{seed}"
                                )
                                # @formatter:on

                                gen_dirs = (
                                    os.path.join(train_dir, "generations_model/eval"),
                                )
                                log_paths = (
                                    os.path.join(train_dir, 'log_gem_sanitize.out'),
                                )

                                for gen_dir, log_path in utils.zip_(gen_dirs, log_paths):
                                    command = (
                                        f"python -m gpt2stuff.eval.dart2gem "
                                        f"--ref_path {ref_path} "
                                        f"--gen_dir {gen_dir} "
                                        f"--task 'convert_gen' "
                                    )
                                    command = wrapper.cpu_job_wrapper(
                                        command,
                                        train_dir=train_dir, conda_env=conda_env, hold_job=False, log_path=log_path,
                                    )
                                    command += '\n'
                                    commands += command

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'sanitize_dart_073121.sh')
        with open(script_path, 'w') as f:
            f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
