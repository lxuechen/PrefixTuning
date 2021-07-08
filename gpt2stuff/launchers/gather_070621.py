"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    0705
purpose:
    Gather the scaling behavior results in `date_0702_v4/`
    Non-private results for scaling law.
notes:
    These are CPU-only jobs. So submit to john.
run:
    python -m gpt2stuff.launchers.gather_070521 --mode "submit"
    python -m gpt2stuff.launchers.gather_070521 --mode "local"
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
                                    f"/nlp/scr/lxuechen/prefixtune/date_0706",
                                    "model_name_distilgpt2_nonprivate_no_"
                                    f"tuning_mode_{tuning_mode}_"
                                    f"per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_"
                                    f"learning_rate_{lr_str}_"
                                    f"train_batch_size_{train_batch_size_str}_"
                                    f"mid_dim_00000512_preseqlen_00000010_"
                                    f"epochs_{epochs_str}_target_epsilon_00000003",
                                    f"{seed}"
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
                                        f"--scratch_dir {scratch_dir}"
                                    )
                                    command = wrapper.cpu_job_wrapper(
                                        command, train_dir=train_dir, conda_env="lxuechen-prefix-tuning",
                                        hold_job=False,
                                        log_path=log_path,
                                    )
                                    command += '\n'
                                    commands += command

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'gather_070621.sh')
        with open(script_path, 'w') as f:
            f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
