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
    noise_multiplier=0.8,
    learning_rate=1e-05,
    # TODO: Artificially truncating since cannot avoid OOM with full + private...
    # TODO: This arg has only been tested for `data2text` so far...
    #   Status for data2text (e2e):
    #   truncate at max_seq_len=96 => training set size 42021
    #   full training set size 42061
    max_seq_len=96,
    max_generations=sys.maxsize,
    objective_mode=0,  # 1 is line-level; 0 is token-level (not suitable with DP).

    eval_steps=100,  # Evaluate every such steps.
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
):
    if mode == Mode.submit and date is None:
        raise ValueError(f"`date` cannot be None when submitting.")

    # Standardize.
    learning_rate_str = wrapper.float2str(learning_rate)
    per_example_max_grad_norm_str = wrapper.float2str(per_example_max_grad_norm)
    noise_multiplier_str = wrapper.float2str(noise_multiplier)
    train_batch_size_str = wrapper.int2str(train_batch_size)

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
                f"/nlp/scr/lxuechen/prefixtune/date_{date}"
                f"/model_name_{model_name_or_path}_nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_per_example_max_grad_norm_{per_example_max_grad_norm_str}_noise_multiplier_{noise_multiplier_str}_learning_rate_{learning_rate_str}_train_batch_size_{train_batch_size_str}"
                f"/{seed}"
            )
            # @formatter:on
        else:
            # @formatter:off
            train_dir = (
                f"/nlp/scr/lxuechen/prefixtune/date_{date}"
                f"/model_name_{model_name_or_path}_nonprivate_{nonprivate}_tuning_mode_{tuning_mode}_learning_rate_{learning_rate_str}_train_batch_size_{train_batch_size_str}"
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
        --noise_multiplier {noise_multiplier} \
        --nonprivate {nonprivate} \
        --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/ \
        --max_steps {max_steps} \
        --max_eval_batches {max_eval_batches} \
        --evaluation_strategy "steps" \
        --per_example_max_grad_norm {per_example_max_grad_norm} \
        --max_seq_len {max_seq_len} \
        --max_generations {max_generations} \
        --train_prompt_file {TRAIN_PROMPT_FILE} \
        --val_prompt_file {VAL_PROMPT_FILE} \
        --eval_prompt_file {EVAL_PROMPT_FILE} \
        --overwrite_output_dir'
    # @formatter:off

    if mode == Mode.submit:
        command = wrapper.mynlprun_wrapper(
            command, train_dir=train_dir, gpu=gpu, conda_env=conda_env, priority=priority
        )
        command += "\n"
    return command
