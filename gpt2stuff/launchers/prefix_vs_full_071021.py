"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    071021
purpose:
    Check exposure!
notes:
    Compare private vs non-private for two repetition levels.
run:
    to generate running scripts:
        python -m gpt2stuff.launchers.prefix_vs_full_071021 --mode "submit"
    to run local:
        python -m gpt2stuff.launchers.prefix_vs_full_071021 --mode "local"
"""

import os

import fire

from .shared import _get_command
from .wrapper import Mode


def main(
    seeds=(0,),  # Seeds over which to randomize.
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

        mid_dim = 512
        preseqlen = 10
        max_eval_batches = 100
        per_device_eval_batch_size = 10
        objective_mode = 0
        eval_epochs = 5
        save_steps = 50000  # So that we don't blow up disk space.
        max_seq_len = 100  # You don't lose too much.
        max_grad_norm = 0.1

        for seed in seeds:
            for model_name_or_path in ("distilgpt2",):
                for nonprivate in ('yes', 'no'):
                    for lr in (5e-4,):
                        for tuning_mode in ("fulltune",):
                            for target_epsilon in (3,):
                                for num_repetitions in (1, 10, 100):

                                    if nonprivate == "yes":
                                        train_batch_size = 5
                                        epochs = 5
                                    else:
                                        train_batch_size = 500
                                        epochs = 50

                                    if model_name_or_path == "distilgpt2":
                                        if tuning_mode == "prefixtune":
                                            per_device_train_batch_size = 40  # Speed up prefix-tuning.
                                        else:
                                            per_device_train_batch_size = 20  # This even fits on regular 12 Gig GPUs.
                                    else:
                                        per_device_train_batch_size = 5  # "gpt2-medium" is large!

                                    if train_batch_size < per_device_train_batch_size:
                                        per_device_train_batch_size = train_batch_size

                                    if tuning_mode in ("fulltune", "scratchtune"):
                                        efficient = "yes"
                                    else:
                                        efficient = "no"

                                    data_folder = (
                                        f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data-{num_repetitions}"
                                    )

                                    commands += _get_command(
                                        date=f"0710-{num_repetitions}",
                                        mode=mode,
                                        script="gpt2stuff.run_language_modeling",

                                        seed=seed,
                                        nonprivate=nonprivate,

                                        eval_epochs=eval_epochs,
                                        evaluation_strategy="epoch",
                                        save_steps=save_steps,
                                        max_eval_batches=max_eval_batches,
                                        per_device_eval_batch_size=per_device_eval_batch_size,

                                        mid_dim=mid_dim,
                                        preseqlen=preseqlen,

                                        model_name_or_path=model_name_or_path,
                                        objective_mode=objective_mode,

                                        # Important hparams.
                                        epochs=epochs,
                                        tuning_mode=tuning_mode,
                                        learning_rate=lr,
                                        target_epsilon=target_epsilon,

                                        # Show small dependence on these.
                                        train_batch_size=train_batch_size,
                                        per_device_train_batch_size=per_device_train_batch_size,
                                        per_example_max_grad_norm=max_grad_norm,

                                        # Ensure no memory issue.
                                        efficient=efficient,
                                        max_seq_len=max_seq_len,

                                        # Faster!
                                        hold_job=True,
                                        priority="low",

                                        data_folder=data_folder,
                                    )

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'prefix_vs_full_071021.sh')
        with open(script_path, 'w') as f:
            f.write(commands)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    fire.Fire(main)
