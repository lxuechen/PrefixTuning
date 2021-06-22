"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    062321
purpose:
    Find best prefix-tuning setup.
notes:
    After fixing the generation and BLEU bugs.
    Also fix the length bug in decoding.
    Train for longer and generate more samples with training set prompts!
    Only train with distilgpt2.
    Refined search.
    More refined learning rate search!
run:
    to generate running scripts:
        python -m gpt2.launchers.prefix_vs_full_062321_v3 --mode "submit"
    to run local:
        python -m gpt2.launchers.prefix_vs_full_062321_v3 --mode "local"
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
            # --- Important bits ---
            for model_name_or_path in ("distilgpt2",):
                epochs = 100
                # --- Important bits ---

                if model_name_or_path == "distilgpt2":
                    per_device_train_batch_size = 25
                else:
                    per_device_train_batch_size = 5  # "gpt2-medium" is large!
                max_grad_norm = 0.1
                noise_multiplier = 0.8
                tuning_mode = "prefixtune"
                mid_dim = 512
                preseqlen = 10
                eval_steps = 500
                max_eval_batches = 100
                per_device_eval_batch_size = 10
                objective_mode = 0
                priority = "standard"
                save_steps = 50000  # So that we don't blow up disk space.
                ema_model_averaging = "no"

                for train_batch_size in (200, 300, 400):
                    for lr in (7e-4,):
                        # 25 is reasonable to fit on a single GPU; but this gives a problem if we want to test out 5.
                        if train_batch_size < per_device_train_batch_size:
                            per_device_train_batch_size = train_batch_size

                        commands += _get_command(
                            date="0623",
                            mode=mode,

                            seed=seed,
                            nonprivate="no",
                            eval_steps=eval_steps,
                            save_steps=save_steps,
                            max_eval_batches=max_eval_batches,
                            per_device_eval_batch_size=per_device_eval_batch_size,

                            per_example_max_grad_norm=max_grad_norm,
                            noise_multiplier=noise_multiplier,
                            tuning_mode=tuning_mode,
                            mid_dim=mid_dim,

                            train_batch_size=train_batch_size,
                            # Roughly 9 Gigs for prefix-tuning.
                            per_device_train_batch_size=per_device_train_batch_size,
                            preseqlen=preseqlen,
                            learning_rate=lr,
                            model_name_or_path=model_name_or_path,
                            objective_mode=objective_mode,
                            priority=priority,
                            ema_model_averaging=ema_model_averaging,

                            epochs=epochs,
                        )

        script_path = os.path.join('.', 'gpt2', 'scripts', f'prefix_vs_full_062321_v3.sh')
        with open(script_path, 'w') as f:
            f.write(commands)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    fire.Fire(main)
