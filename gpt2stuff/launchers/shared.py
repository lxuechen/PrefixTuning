import sys

from . import wrapper
from .wrapper import Mode

TRAIN_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt"
VAL_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt"
EVAL_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt"

TRAIN_PROMPT_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_train.txt"
VAL_PROMPT_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_valid.txt"
EVAL_PROMPT_FILE = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_test.txt"


def _get_command(
    seed,
    tuning_mode,
    nonprivate,
    date=None,  # Always include this so as to not mess up the folders.

    # Don't modify these easily!
    epochs=5,
    train_batch_size=5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=1,
    per_example_max_grad_norm=1.,
    noise_multiplier=-1,
    learning_rate=1e-05,
    # TODO: Artificially truncating since cannot avoid OOM with full + private...
    # TODO: This arg has only been tested for `data2text` so far...
    #   Status for data2text (e2e):
    #   truncate at max_seq_len=96 => training set size 42021
    #   full training set size 42061
    max_seq_len=96,
    max_generations=sys.maxsize,
    max_generations_train=60,
    objective_mode=0,  # 1 is line-level; 0 is token-level (not suitable with DP).

    eval_steps=100,  # Evaluate every such steps.
    eval_epochs=10,
    max_steps=-1,
    max_eval_batches=-1,
    mid_dim=512,
    preseqlen=5,
    save_steps=500000,
    mode="submit",
    model_type="gpt2",
    model_name_or_path="distilgpt2",  # 80+million
    gpu=None,  # Randomly grab.
    conda_env="lxuechen-prefix-tuning",
    priority="standard",
    time=None,

    script="gpt2stuff.run_language_modeling",
    train_dir=None,
    ema_model_start_from=1000,
    ema_model_averaging="no",
    efficient="no",
    debug="no",
    evaluation_strategy="epoch",

    # -1 is just a default value.
    target_epsilon=-1,
    target_delta=-1,
    task_mode="data2text",
    hold_job=True,
    lr_decay="yes",
):
    if mode == Mode.submit and date is None:
        raise ValueError(f"`date` cannot be None when submitting.")

    # TODO: Fix a delta for each dataset!
    if target_delta < 0:
        if task_mode == "data2text":
            target_delta = 1e-5
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")

    # Standardize.
    learning_rate_str = wrapper.float2str(learning_rate)
    per_example_max_grad_norm_str = wrapper.float2str(per_example_max_grad_norm)
    noise_multiplier_str = wrapper.float2str(noise_multiplier)
    train_batch_size_str = wrapper.int2str(train_batch_size)
    mid_dim_str = wrapper.int2str(mid_dim)
    preseqlen_str = wrapper.int2str(preseqlen)
    epochs_str = wrapper.int2str(epochs)
    target_epsilon_str = wrapper.int2str(target_epsilon)

    # Check mode.
    if mode == Mode.submit:
        gradient_accumulation_steps = train_batch_size // per_device_train_batch_size

        if nonprivate == "no":
            if train_dir is None:
                # @formatter:off
                train_dir = (
                    f"/nlp/scr/lxuechen/prefixtune/date_{date}"
                    f"/model_name_{model_name_or_path}_nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_per_example_max_grad_norm_{per_example_max_grad_norm_str}_noise_multiplier_{noise_multiplier_str}_learning_rate_{learning_rate_str}_train_batch_size_{train_batch_size_str}_mid_dim_{mid_dim_str}_preseqlen_{preseqlen_str}_epochs_{epochs_str}_target_epsilon_{target_epsilon_str}"
                    f"/{seed}"
                )
                # @formatter:on
        else:
            if train_dir is None:
                # @formatter:off
                train_dir = (
                    f"/nlp/scr/lxuechen/prefixtune/date_{date}"
                    f"/model_name_{model_name_or_path}_nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_learning_rate_{learning_rate_str}_train_batch_size_{train_batch_size_str}_mid_dim_{mid_dim_str}_preseqlen_{preseqlen_str}_epochs_{epochs_str}_target_epsilon_{target_epsilon_str}"
                    f"/{seed}"
                )
                # @formatter:on
    else:
        if train_dir is None:
            # Local debugging.
            train_dir = "/nlp/scr/lxuechen/tests/prefix-tuning"

    # @formatter:off
    logging_dir = train_dir
    command = f'python -m {script} \
        --output_dir {train_dir} \
        --task_mode {task_mode} \
        --model_type {model_type} \
        --model_name_or_path {model_name_or_path} \
        --tokenizer_name {model_name_or_path} \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --per_device_eval_batch_size {per_device_eval_batch_size} \
        --save_steps {save_steps} \
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
        --eval_epochs {eval_epochs} \
        --noise_multiplier {noise_multiplier} \
        --nonprivate {nonprivate} \
        --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/ \
        --max_steps {max_steps} \
        --max_eval_batches {max_eval_batches} \
        --evaluation_strategy {evaluation_strategy} \
        --per_example_max_grad_norm {per_example_max_grad_norm} \
        --max_seq_len {max_seq_len} \
        --max_generations {max_generations} \
        --max_generations_train {max_generations_train} \
        --train_prompt_file {TRAIN_PROMPT_FILE} \
        --val_prompt_file {VAL_PROMPT_FILE} \
        --eval_prompt_file {EVAL_PROMPT_FILE} \
        --ema_model_averaging {ema_model_averaging} \
        --ema_model_start_from {ema_model_start_from} \
        --efficient {efficient} \
        --debug {debug} \
        --target_delta {target_delta} \
        --target_epsilon {target_epsilon} \
        --overwrite_output_dir \
        --lr_decay {lr_decay}'
    # @formatter:off

    if mode == Mode.submit:
        command = wrapper.mynlprun_wrapper(
            command,
            train_dir=train_dir,
            gpu=gpu, conda_env=conda_env, priority=priority, time=time, hold_job=hold_job,
        )
        command += "\n"
    return command
