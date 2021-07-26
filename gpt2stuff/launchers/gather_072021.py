"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    0720
purpose:
    Gather DART results.
notes:
    These are CPU-only jobs. So submit to john.
run:
    python -m gpt2stuff.launchers.gather_072021 --mode "submit"
    python -m gpt2stuff.launchers.gather_072021 --mode "local"
"""

import os
import uuid

import fire

from lxuechen_utils import utils
from . import wrapper


def main(
    seeds=(0,),
    mode=wrapper.Mode.local,
):
    if mode == wrapper.Mode.local:
        pass
    elif mode == wrapper.Mode.submit:
        commands = ""

        max_grad_norm = 0.1
        noise_multiplier = -1
        ref_path = "/nlp/scr/lxuechen/data/prefix-tuning/data/dart/clean_references_test.txt"

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
                                        f"nonprivate_no_"
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
                                        os.path.join(train_dir, "generations_ema_model/eval"),
                                    )
                                    img_dirs = (
                                        os.path.join(train_dir, 'generations_score'),
                                        os.path.join(train_dir, 'generations_ema_score'),
                                    )
                                    log_paths = (
                                        os.path.join(train_dir, 'log_cpu.out'),
                                        os.path.join(train_dir, 'log_ema_cpu.out'),
                                    )

                                    for gen_dir, img_dir, log_path in utils.zip_(gen_dirs, img_dirs, log_paths):
                                        scratch_dir = f"/nlp/scr/lxuechen/scratch/tmp/{str(uuid.uuid4())}"
                                        command = (
                                            f"python -m gpt2stuff.eval.eval_generations "
                                            f"--task eval_trajectory --gen_dir {gen_dir} --img_dir {img_dir} "
                                            f"--scratch_dir {scratch_dir} --ref_path {ref_path}"
                                        )
                                        command = wrapper.cpu_job_wrapper(
                                            command, train_dir=train_dir, conda_env="lxuechen-prefix-tuning",
                                            hold_job=False,
                                            log_path=log_path,
                                        )
                                        command += '\n'
                                        commands += command

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'gather_072021.sh')
        with open(script_path, 'w') as f:
            f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
