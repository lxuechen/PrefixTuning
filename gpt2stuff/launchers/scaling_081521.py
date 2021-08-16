"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    081521
purpose:
    New scaling experiments with more evaluation.
    Still with aspect ratio 32.

    Deeper models.
notes:
    Fix the failed jobs!!!

    Frequently check if your launcher is still running:
        ps ax | grep gpt2stuff.launchers.scaling
run:
    Local:
        sc1

    On server:
        python -m gpt2stuff.launchers.scaling_081521
"""

import os
import subprocess
import time

import GPUtil
import fire

# Get the paths to checkpoints.
aspect_ratio = 32  # d_model / n_layer (distilgpt2=128)
n_layers = tuple(range(2, 32, 2))
pretrained_folders = ()
for n_layer in n_layers:
    d_model = int(n_layer * aspect_ratio)
    pretrained_folders += (
        f"/home/lxuechen_stanford_edu/lxuechen-buck/distilgpt2/date_080221/distilgpt2-{n_layer}-{d_model}",
    )
del n_layer


def _get_command(
    gpu_id,
    target_epsilon,
    pretrained_folder,
    train_dir,
    tuning_mode,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    learning_rate,
    epochs,
    eval_epochs,
    seed,
    max_generations_train=10,
):
    command = f"mkdir -p {train_dir} \n"
    command += f'CUDA_VISIBLE_DEVICES={gpu_id} python -m gpt2stuff.launchers.prefix_vs_full_062021 ' \
               f'--mode "local" ' \
               f'--tuning_mode {tuning_mode} ' \
               f'--max_seq_len 100 ' \
               f'--nonprivate "no" ' \
               f'--per_device_train_batch_size {per_device_train_batch_size} ' \
               f'--gradient_accumulation_steps {gradient_accumulation_steps} ' \
               f'--learning_rate {learning_rate} ' \
               f'--per_example_max_grad_norm 0.1 ' \
               f'--target_epsilon {target_epsilon} ' \
               f'--epochs {epochs} ' \
               f'--eval_epochs {eval_epochs} ' \
               f'--private_engine_mode "ghost"   ' \
               f'--model_name_or_path "{pretrained_folder}" ' \
               f'--max_generations_train {max_generations_train} ' \
               f'--seed {seed} ' \
               f'--train_dir {train_dir} & '
    return command + '\n'


def main(
    num_gpus=8,

    train_batch_size=512,
    per_device_batch_size=32,
    target_epsilon=8,
    epochs=50,
    eval_epochs=5,
    learning_rate=5e-4,
    date="081521",
    seeds=(1, 2),
):
    gradient_accumulation_steps = train_batch_size // per_device_batch_size

    job_id = 0

    for seed in seeds:
        for tuning_mode in ("fulltune", 'scratchtune'):
            for n_layer, pretrained_folder in zip(n_layers, pretrained_folders):
                if n_layer in list(range(2, 18, 2)) and tuning_mode == "fulltune":
                    continue

                empty_gpus = []
                while len(empty_gpus) == 0:
                    # Notes: getFirstAvailable is very bad since it throws RuntimeError when no GPU is found.
                    empty_gpus = GPUtil.getAvailable(
                        order='first',
                        maxLoad=0.001,
                        maxMemory=0.001,
                        limit=1,
                    )
                    time.sleep(1)
                print(f'empty gpus: {empty_gpus}')
                gpu_id = empty_gpus[0]

                base_name = os.path.basename(pretrained_folder)
                train_dir = f"/nlp/scr/lxuechen/prefixtune/date_081421/{tuning_mode}/{base_name}/{seed}"
                command = _get_command(
                    gpu_id=gpu_id,
                    target_epsilon=target_epsilon,
                    pretrained_folder=pretrained_folder,
                    train_dir=train_dir,
                    tuning_mode=tuning_mode,
                    per_device_train_batch_size=per_device_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    eval_epochs=eval_epochs,
                    seed=seed,
                )
                # os.system(command)  # This stupid thing waits!!!

                # This doesn't wait.
                subprocess.Popen(
                    [command],
                    shell=True, stdin=None, stdout=None, stderr=None, close_fds=True
                )

                # Give the program some time to be located on the GPU, before scheduling the next.
                time.sleep(240)
                print(f'scheduling job: {job_id} on gpu: {gpu_id}')

                job_id += 1


if __name__ == "__main__":
    fire.Fire(main)
