"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    080521
purpose:
    Scale model size experiment for figure 2.
notes:
    fulltune first, then scratchtune.
    Patch the missing experiment where n_layer=16.
run:
    python -m gpt2stuff.launchers.scaling_080521
"""

import os

import fire

# Get the paths to checkpoints.
aspect_ratio = 32  # d_model / n_layer (distilgpt2=128)
n_layers = (16,)
pretrained_folders = ()
for n_layer in n_layers:
    d_model = int(n_layer * aspect_ratio)
    pretrained_folders += (
        f"/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-{n_layer}-{d_model}",
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
    nonprivate="no",
):
    command = f"mkdir -p {train_dir} \n"
    command += f'CUDA_VISIBLE_DEVICES={gpu_id} python -m gpt2stuff.launchers.prefix_vs_full_062021 ' \
               f'--mode "local" ' \
               f'--nonprivate {nonprivate} ' \
               f'--tuning_mode {tuning_mode} ' \
               f'--max_seq_len 100 ' \
               f'--per_device_train_batch_size {per_device_train_batch_size} ' \
               f'--gradient_accumulation_steps {gradient_accumulation_steps} ' \
               f'--learning_rate {learning_rate} ' \
               f'--per_example_max_grad_norm 0.1 ' \
               f'--target_epsilon {target_epsilon} ' \
               f'--epochs {epochs} ' \
               f'--private_engine_mode "ghost"   ' \
               f'--model_name_or_path "{pretrained_folder}" ' \
               f'--train_dir {train_dir} |& tee {train_dir}/log.out & '
    return command + '\n'


def main(
    num_gpus=1,
    gpu_init_id=7,
):
    commands = "#!/bin/bash\n"
    gpu_id = gpu_init_id

    date = "080321"
    train_batch_size = 512
    per_device_batch_size = 32
    target_epsilon = 8
    epochs = 50
    learning_rate = 5e-4
    gradient_accumulation_steps = train_batch_size // per_device_batch_size
    nonprivate = "no"

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
                nonprivate=nonprivate,
            )
            commands += command
            gpu_id += 1
            if (gpu_id - gpu_init_id) % num_gpus == 0:
                gpu_id = gpu_init_id
                commands += 'wait\n'

    date = "080421"
    train_batch_size = 5
    per_device_batch_size = 5
    target_epsilon = 8
    epochs = 5
    learning_rate = 5e-5
    gradient_accumulation_steps = train_batch_size // per_device_batch_size
    nonprivate = "yes"

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
                nonprivate=nonprivate,
            )
            commands += command
            gpu_id += 1
            if (gpu_id - gpu_init_id) % num_gpus == 0:
                gpu_id = gpu_init_id
                commands += 'wait\n'

    script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'scaling_080521.sh')
    with open(script_path, 'w') as f:
        f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
