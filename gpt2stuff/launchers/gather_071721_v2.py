"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    0717
purpose:
    Gather varying batch size, fixed learning rate experiments.
notes:
    These are CPU-only jobs. So submit to john.
run:
    python -m gpt2stuff.launchers.gather_071721_v2 --mode "submit"
    python -m gpt2stuff.launchers.gather_071721_v2 --mode "local"
"""

import os
import uuid

import fire

from lxuechen_utils import utils
from . import wrapper


def main(
    seeds=(1,),
    mode=wrapper.Mode.local,
):
    if mode == wrapper.Mode.local:
        pass
    elif mode == wrapper.Mode.submit:
        commands = ""

        target_epsilon = -1
        for seed in seeds:
            for max_grad_norm in (0.1,):
                for model_name_or_path in ("distilgpt2",):
                    for i, train_batch_size in enumerate((1024,)):
                        for lr in (1e-3, 5e-4, 1e-4,):
                            for noise_multiplier in (0.01, 0.05, 0.1, 0.5, 1):
                                for epochs in (50,):
                                    for tuning_mode in ("prefixtune", "fulltune"):

                                        epochs_str = utils.int2str(epochs)
                                        lr_str = utils.float2str(lr)
                                        target_epsilon_str = utils.int2str(target_epsilon)
                                        train_batch_size_str = utils.int2str(train_batch_size)
                                        max_grad_norm_str = utils.float2str(max_grad_norm)
                                        noise_multiplier_str = utils.float2str(noise_multiplier)

                                        # @formatter:off
                                        train_dir = (
                                            f"/nlp/scr/lxuechen/prefixtune/date_0717"
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
                                                f"--scratch_dir {scratch_dir}"
                                            )
                                            command = wrapper.cpu_job_wrapper(
                                                command, train_dir=train_dir, conda_env="lxuechen-prefix-tuning",
                                                hold_job=False,
                                                log_path=log_path,
                                            )
                                            command += '\n'
                                            commands += command

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'gather_071721_v2.sh')
        with open(script_path, 'w') as f:
            f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
