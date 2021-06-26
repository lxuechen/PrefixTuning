"""Generate shell scripts to run.

This is a template script. Filling out the stuff below each time before running
new experiments helps with organization and avoids the messiness once there are
many experiments.

date:
    062621
purpose:
    Debug cublas fail error in 062621.
notes:
    The problem seems more dependent on the GPU and machine.
    This was not ran, since I already figured out it's because PyTorch is evil.
    https://github.com/allenai/allennlp/issues/5064
    https://github.com/pytorch/pytorch/issues/53957
run:
    to generate running scripts:
        python -m gpt2.launchers.prefix_vs_full_062621_v2 --mode "submit"
    to run local:
        python -m gpt2.launchers.prefix_vs_full_062621_v2 --mode "local"
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
    gpu="titanx",
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

                for target_epsilon in (8,):
                    for epochs in (60,):
                        for lr in (5e-4,):
                            for tuning_mode in ("scratchtune",):

                                if model_name_or_path == "distilgpt2":
                                    per_device_train_batch_size = 20  # This even fits on regular 12 Gig GPUs.
                                else:
                                    per_device_train_batch_size = 5  # "gpt2-medium" is large!

                                mid_dim = 512
                                preseqlen = 10
                                max_eval_batches = 100
                                per_device_eval_batch_size = 10
                                objective_mode = 0
                                priority = "standard"
                                eval_steps = 1000
                                save_steps = 1000  # So that we don't blow up disk space.
                                max_seq_len = 100  # You don't lose too much.

                                # TODO: Show small dependence on these parameters!
                                max_grad_norm = 0.1
                                train_batch_size = 400

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
                                    nonprivate="no",
                                    eval_steps=eval_steps,
                                    save_steps=save_steps,
                                    max_eval_batches=max_eval_batches,
                                    per_device_eval_batch_size=per_device_eval_batch_size,

                                    mid_dim=mid_dim,
                                    preseqlen=preseqlen,

                                    model_name_or_path=model_name_or_path,
                                    objective_mode=objective_mode,
                                    priority=priority,

                                    # Important hparams.
                                    epochs=epochs,
                                    tuning_mode=tuning_mode,
                                    target_epsilon=target_epsilon,
                                    learning_rate=lr,

                                    # Show small dependence on these.
                                    train_batch_size=train_batch_size,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    per_example_max_grad_norm=max_grad_norm,

                                    # Ensure no memory issue.
                                    efficient=efficient,
                                    max_seq_len=max_seq_len,
                                    gpu=gpu,
                                )

        script_path = os.path.join('.', 'gpt2', 'scripts', f'prefix_vs_full_062621_v2.sh')
        with open(script_path, 'w') as f:
            f.write(commands)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    fire.Fire(main)

# @formatter:off
# Fix error here!
# model_name_distilgpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_80000000_learning_rate_0_00050000_train_batch_size_00000400_mid_dim_00000512_preseqlen_00000010_epochs_00000060_target_epsilon_00000008/0/log.out
# @formatter:on
