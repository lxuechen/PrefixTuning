"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    0803
purpose:
    Gather new results for better non-private models.
notes:
    These are CPU-only jobs. So submit to john.
run:
    python -m gpt2stuff.launchers.gather_081021 --mode "submit"
    python -m gpt2stuff.launchers.gather_081021 --mode "local"
"""

import os
import uuid

import fire

from lxuechen_utils import utils
from . import wrapper


def main(
    base_dir=f"/nlp/scr/lxuechen/prefixtune/date_0721",
    seeds=(0,),
    mode=wrapper.Mode.local,
):
    if mode == wrapper.Mode.local:
        pass
    elif mode == wrapper.Mode.submit:
        commands = ""

        max_grad_norm = 0.1
        noise_multiplier = -1

        nonprivate = "yes"
        for seed in seeds:
            for model_name_or_path in ("gpt2", "gpt2-medium"):
                for tuning_mode in ("prefixtune", "fulltune", "scratchtune", "lineartune",):
                    for train_batch_size in (5,):
                        for epochs in (10,):
                            for lr in (5e-5,):
                                for _ in ("",):
                                    epochs_str = utils.int2str(epochs)
                                    lr_str = utils.float2str(lr)
                                    train_batch_size_str = utils.int2str(train_batch_size)

                                    # @formatter:off
                                    train_dir = os.path.join(
                                        base_dir,
                                        f"model_name_{model_name_or_path}_"
                                        f"nonprivate_{nonprivate}_"
                                        f"tuning_mode_{tuning_mode}_"
                                        f"learning_rate_{lr_str}_"
                                        f"train_batch_size_{train_batch_size_str}_"
                                        "mid_dim_00000512_"
                                        "preseqlen_00000010_"
                                        f"epochs_{epochs_str}_"
                                        "target_epsilon_-0000001",
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

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'gather_081021.sh')
        with open(script_path, 'w') as f:
            f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
