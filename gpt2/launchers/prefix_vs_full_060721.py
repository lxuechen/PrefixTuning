"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    060721
purpose:
    Compare prefix-tuning and full-tuning.
notes:
    NA
run:
    python -m gpt2.launchers.prefix_vs_full_060721
"""

import os

import fire

from . import wrapper

TRAIN_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt"
TEST_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt"


def _get_command(
    seed,
    train_dir,
    epochs,
    tuning_mode,
    nonprivate,

    # Don't modify these easily!
    eval_steps=100,
    per_device_train_batch_size=5,
    noise_multiplier=1,
    max_steps=-1,
    max_eval_steps=-1,
    learning_rate=5e-05,
    mid_dim=512,
    preseqlen=5,
    mode="submit",
    model_type="gpt2",
    model_name_or_path="distilgpt2",  # 80+million
):
    # @formatter:off
    logging_dir = train_dir
    command = f'python -m gpt2.run_language_modeling \
        --output_dir {train_dir} \
        --task_mode "data2text" \
        --model_type {model_type} \
        --model_name_or_path {model_name_or_path} \
        --tokenizer_name {model_name_or_path} \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --per_device_eval_batch_size 10 \
        --save_steps 500000 \
        --num_train_epochs {epochs} \
        --do_train \
        --do_eval \
        --line_by_line \
        --save_total_limit 1 \
        --train_data_file {TRAIN_FILE} \
        --eval_data_file {TEST_FILE} \
        --tuning_mode {tuning_mode} \
        --logging_dir {logging_dir} \
        --optim_prefix yes \
        --preseqlen {preseqlen} \
        --prefix_mode activation \
        --format_mode cat \
        --gradient_accumulation_steps 1 \
        --learning_rate {learning_rate} \
        --weight_decay 0.0 \
        --seed {seed} \
        --mid_dim {mid_dim} \
        --init_random no \
        --use_dropout no \
        --prefix_dropout 0.0 \
        --objective_mode 1 \
        --evaluate_during_training \
        --eval_steps {eval_steps} \
        --noise_multiplier {noise_multiplier} \
        --nonprivate {nonprivate} \
        --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/ \
        --max_steps {max_steps} \
        --max_eval_steps {max_eval_steps} \
        --evaluation_strategy "steps" \
        --overwrite_output_dir'
    # @formatter:off
    # TODO: Fix eval_steps
    if mode == "submit":
        command = wrapper.mynlprun_wrapper(command, train_dir=train_dir)
        command += "\n"
    return command


def main(
    seeds=(0, 1, 2),  # Seeds over which to randomize.
    mode="local",

    # For local testing; don't modify these defaults!
    tuning_mode="prefixtune",
    nonprivate="yes",
    max_steps=1,
    max_eval_steps=20,

    max_jobs_in_queue=10,  # Number of jobs in each batch.
    sleep_seconds=3600,  # Seconds to sleep before launching the next batch of jobs.
    jobs_in_queue=0,  # Number of jobs in the queue.
    **kwargs,
):
    if mode == "local":
        command = _get_command(
            seed=0,
            train_dir="/nlp/scr/lxuechen/tests/prefix-tuning",
            epochs=1,
            tuning_mode=tuning_mode,
            mode=mode,
            max_steps=max_steps,
            max_eval_steps=max_eval_steps,
            nonprivate=nonprivate,
            **kwargs,
        )
        print(command)
        os.system(command)

    elif mode == "submit":
        commands = "#!/bin/bash\n"
        for seed in seeds:
            # TODO: 3090, titanrtx,
            # TODO: 1) Gradient accumulation for private full, 2) private full needs better GPUs
            pass

        script_path = os.path.join('.', 'gpt2', 'scripts', f'prefix_vs_full_060721.sh')
        with open(script_path, 'w') as f:
            f.write(commands)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    fire.Fire(main)
