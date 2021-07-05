"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    070421
purpose:
    Run lineartune baseline on E2E.
notes:
    Patches experiments from 062621
    This is the non-private version.
run:
    to generate running scripts:
        python -m gpt2stuff.launchers.prefix_vs_full_070421_v2 --mode "submit"
    to run local:
        python -m gpt2stuff.launchers.prefix_vs_full_070421_v2 --mode "local"
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

        for seed in seeds:
            for model_name_or_path in ("distilgpt2",):
                for epochs in (5,):
                    for lr in (5e-5,):
                        for tuning_mode in ("lineartune",):

                            if model_name_or_path == "distilgpt2":
                                per_device_train_batch_size = 20  # This even fits on regular 12 Gig GPUs.
                            else:
                                per_device_train_batch_size = 5  # "gpt2-medium" is large!

                            mid_dim = 512
                            preseqlen = 10
                            max_eval_batches = 100
                            per_device_eval_batch_size = 10
                            objective_mode = 0
                            eval_steps = 5000
                            save_steps = 50000  # So that we don't blow up disk space.
                            max_seq_len = 100  # You don't lose too much.

                            # TODO: Show small dependence on these parameters!
                            max_grad_norm = 0.1
                            train_batch_size = 5

                            if train_batch_size < per_device_train_batch_size:
                                per_device_train_batch_size = train_batch_size

                            if tuning_mode in ("fulltune", "scratchtune"):
                                efficient = "yes"
                            else:
                                efficient = "no"

                            commands += _get_command(
                                date="0626",
                                mode=mode,

                                seed=seed,
                                nonprivate="yes",
                                eval_steps=eval_steps,
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
                            )

        script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'prefix_vs_full_070421_v2.sh')
        with open(script_path, 'w') as f:
            f.write(commands)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    fire.Fire(main)
