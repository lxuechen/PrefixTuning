"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    081221
purpose:
    New scaling experiments with more evaluation.
    Still with aspect ratio 32.

    Deeper models.
notes:
    fulltune first, then scratchtune.
run:
    python -m gpt2stuff.launchers.scaling_081221_v2 ; sc1
"""

import os

import fire

# Get the paths to checkpoints.
aspect_ratio = 32  # d_model / n_layer (distilgpt2=128)
n_layers = (24, 26, 28, 30,)
pretrained_folders = ()
# TODO: This is different since I made a stupid mistake.
for n_layer in n_layers:
    d_model = int(n_layer * aspect_ratio)
    pretrained_folders += (
        f"/home/lxuechen_stanford_edu/dump/distilgpt2/date_080210/distilgpt2-{n_layer}-{d_model}",
    )


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
    max_generations_train=20,
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
               f'--train_dir {train_dir} |& tee {train_dir}/log.out & '
    return command + '\n'


def main(

    train_batch_size=512,
    per_device_batch_size=32,
    target_epsilon=8,
    epochs=50,
    eval_epochs=5,
    learning_rate=5e-4,
    date="081221",

    init_gpu_id=3,
    num_gpus=4,
):
    commands = "#!/bin/bash\n"
    gpu_id = init_gpu_id
    gradient_accumulation_steps = train_batch_size // per_device_batch_size

    # Private finetune.
    for tuning_mode in ("fulltune", 'scratchtune'):
        for pretrained_folder in pretrained_folders:
            base_name = os.path.basename(pretrained_folder)
            train_dir = f"/nlp/scr/lxuechen/prefixtune/date_{date}/{tuning_mode}/{base_name}"
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
            )
            commands += command
            gpu_id += 1
            if (gpu_id - init_gpu_id) % num_gpus == 0:
                gpu_id = init_gpu_id
                commands += 'wait\n'

    script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'scaling_081221_v2.sh')
    with open(script_path, 'w') as f:
        f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
