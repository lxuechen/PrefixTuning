"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    061721
purpose:
    Check out the non-private generations!
notes:
    NA
run:
    to generate running scripts:
        python -m gpt2.launchers.prefix_vs_full_061721_v2 --mode "submit"
    to run local:
        python -m gpt2.launchers.prefix_vs_full_061721_v2 --mode "local"
"""

import os
import sys

import fire

from . import wrapper
from .wrapper import Mode

TRAIN_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt"
VAL_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt"
EVAL_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt"


def _get_command(
    seed,
    tuning_mode,
    nonprivate,

    # Don't modify these easily!
    epochs=5,
    train_batch_size=5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=1,
    per_example_max_grad_norm=1.,
    noise_multiplier=0.8,
    learning_rate=1e-05,
    # TODO: Artificially truncating since cannot avoid OOM with full + private...
    # TODO: This arg has only been tested for `data2text` so far...
    #   Status for data2text (e2e):
    #   truncate at max_seq_len=96 => training set size 42021
    #   full training set size 42061
    max_seq_len=96,
    max_generations=sys.maxsize,
    objective_mode=1,  # 1 is line-level; 0 is token-level (not suitable with DP).

    eval_steps=100,  # Evaluate every such steps.
    max_steps=-1,
    max_eval_batches=-1,
    mid_dim=512,
    preseqlen=5,
    mode="submit",
    model_type="gpt2",
    model_name_or_path="distilgpt2",  # 80+million
    gpu=None,  # Randomly grab.
    conda_env="lxuechen-prefix-tuning"
):
    # Standardize.
    learning_rate_str = wrapper.float2str(learning_rate)
    per_example_max_grad_norm_str = wrapper.float2str(per_example_max_grad_norm)
    noise_multiplier_str = wrapper.float2str(noise_multiplier)
    train_batch_size_str = wrapper.int2str(train_batch_size)
    mid_dim_str = wrapper.int2str(mid_dim)

    # Check mode.
    if mode == Mode.submit:
        if tuning_mode == "fulltune" and nonprivate == "no":
            # TODO: This disables setting batch size from the outside.
            per_device_train_batch_size = 1
            gradient_accumulation_steps = 5
            gpu = "3090"  # This stupid thing needs a lot of memory!!!
        else:
            gradient_accumulation_steps = train_batch_size // per_device_train_batch_size

        if nonprivate == "no":
            # @formatter:off
            train_dir = (
                f"/nlp/scr/lxuechen/prefixtune/date_0617"
                f"/model_name_{model_name_or_path}_nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_per_example_max_grad_norm_{per_example_max_grad_norm_str}_noise_multiplier_{noise_multiplier_str}_learning_rate_{learning_rate_str}_train_batch_size_{train_batch_size_str}"
                f"/{seed}"
            )
            # @formatter:on
        else:
            # @formatter:off
            train_dir = (
                f"/nlp/scr/lxuechen/prefixtune/date_0617"
                f"/model_name_{model_name_or_path}_nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_learning_rate_{learning_rate_str}_train_batch_size_{train_batch_size_str}_mid_dim_{mid_dim_str}"
                f"/{seed}"
            )
            # @formatter:on
    else:
        # Local debugging.
        train_dir = "/nlp/scr/lxuechen/tests/prefix-tuning"

    # @formatter:off
    logging_dir = train_dir
    command = f'python -m gpt2.run_language_modeling \
        --output_dir {train_dir} \
        --task_mode "data2text" \
        --model_type {model_type} \
        --model_name_or_path {model_name_or_path} \
        --tokenizer_name {model_name_or_path} \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --per_device_eval_batch_size {per_device_eval_batch_size} \
        --save_steps 500000 \
        --num_train_epochs {epochs} \
        --do_train \
        --do_eval \
        --line_by_line \
        --save_total_limit 1 \
        --train_data_file {TRAIN_FILE} \
        --val_data_file {VAL_FILE} \
        --eval_data_file {EVAL_FILE} \
        --tuning_mode {tuning_mode} \
        --logging_dir {logging_dir} \
        --logging_steps -1 \
        --optim_prefix yes \
        --preseqlen {preseqlen} \
        --prefix_mode activation \
        --format_mode cat \
        --gradient_accumulation_steps {gradient_accumulation_steps} \
        --learning_rate {learning_rate} \
        --weight_decay 0.0 \
        --seed {seed} \
        --mid_dim {mid_dim} \
        --init_random no \
        --use_dropout no \
        --prefix_dropout 0.0 \
        --objective_mode {objective_mode} \
        --evaluate_during_training \
        --eval_steps {eval_steps} \
        --noise_multiplier {noise_multiplier} \
        --nonprivate {nonprivate} \
        --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/ \
        --max_steps {max_steps} \
        --max_eval_batches {max_eval_batches} \
        --evaluation_strategy "steps" \
        --per_example_max_grad_norm {per_example_max_grad_norm} \
        --max_seq_len {max_seq_len} \
        --max_generations {max_generations} \
        --overwrite_output_dir'
    # @formatter:off

    if mode == Mode.submit:
        command = wrapper.mynlprun_wrapper(command, train_dir=train_dir, gpu=gpu, conda_env=conda_env)
        command += "\n"
    return command


def main(
    seeds=(0, ),  # Seeds over which to randomize.
    mode=Mode.local,

    # For local testing; don't modify these defaults!
    tuning_mode="prefixtune",
    nonprivate="yes",
    max_steps=1,
    max_eval_batches=20,

    max_jobs_in_queue=10,  # Number of jobs in each batch.
    sleep_seconds=3600,  # Seconds to sleep before launching the next batch of jobs.
    jobs_in_queue=0,  # Number of jobs in the queue.
    **kwargs,
):
    if mode == Mode.local:
        command = _get_command(
            seed=0,
            epochs=1,
            tuning_mode=tuning_mode,
            mode=mode,
            max_steps=max_steps,
            max_eval_batches=max_eval_batches,
            nonprivate=nonprivate,
            **kwargs,
        )
        print(command)
        os.system(command)

    elif mode == Mode.submit:
        # Goal is to check out the nonprivate generations!
        commands = "#!/bin/bash\n"

        for seed in seeds:
            train_batch_size = 5
            per_device_train_batch_size = 5
            mid_dim = 256
            preseqlen = 5
            nonprivate = "yes"

            for tuning_mode in ("prefixtune", "fulltune"):
                for lr in (1e-5, 1e-6):
                    # 25 is reasonable to fit on a single GPU; but this gives a problem if we want to test out 5.
                    if train_batch_size < per_device_train_batch_size:
                        per_device_train_batch_size = train_batch_size

                    commands += _get_command(
                        mode=mode,

                        seed=seed,
                        nonprivate=nonprivate,
                        eval_steps=100,
                        max_eval_batches=100,
                        per_device_eval_batch_size=25,

                        tuning_mode=tuning_mode,
                        mid_dim=mid_dim,

                        train_batch_size=train_batch_size,
                        # Roughly 9 Gigs for prefix-tuning.
                        per_device_train_batch_size=per_device_train_batch_size,
                        preseqlen=preseqlen,
                        learning_rate=lr,
                    )

        script_path = os.path.join('.', 'gpt2', 'scripts', f'prefix_vs_full_061721_v2.sh')
        with open(script_path, 'w') as f:
            f.write(commands)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    fire.Fire(main)
