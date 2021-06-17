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
                        priority="high",
                    )

        script_path = os.path.join('.', 'gpt2', 'scripts', f'prefix_vs_full_061721_v2.sh')
        with open(script_path, 'w') as f:
            f.write(commands)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    fire.Fire(main)
