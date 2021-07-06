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
    seed=0,
    mode=wrapper.Mode.local,
):
    if mode == wrapper.Mode.local:
        pass
    elif mode == wrapper.Mode.submit:
        commands = ""

        for model_name_or_path in ("distilgpt2", "gpt2", "gpt2-medium"):
            # @formatter:off
            train_dir = os.path.join(
                f"/nlp/scr/lxuechen/prefixtune/date_0702_v4",
                f"model_name_{model_name_or_path}_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001",
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
                    command, train_dir=train_dir, conda_env="lxuechen-prefix-tuning", hold_job=False,
                    log_path=log_path,
                )
                command += '\n'
                commands += command

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'gather_070521.sh')
        with open(script_path, 'w') as f:
            f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
