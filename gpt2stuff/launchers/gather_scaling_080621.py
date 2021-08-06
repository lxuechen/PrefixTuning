# python -m gpt2stuff.launchers.gather_scaling_080621
import os
import uuid

import fire

# Get the paths to checkpoints.
aspect_ratio = 32  # d_model / n_layer (distilgpt2=128)
n_layers = (2, 4, 6, 8, 10, 12, 14,)
base_names = ()
for n_layer in n_layers:
    d_model = int(n_layer * aspect_ratio)
    base_names += (f"distilgpt2-{n_layer}-{d_model}",)


def main(
    scratch_base="/home/lxuechen_stanford_edu/scratch/tmp",
):
    commands = ""

    commands += f"mkdir -p {scratch_base}\n"

    for base_dir in (
        "/nlp/scr/lxuechen/prefixtune/date_080321",
        "/nlp/scr/lxuechen/prefixtune/date_080421"
    ):
        for tuning_mode in ("fulltune", "scratchtune"):
            for base_name in base_names:
                this_dir = os.path.join(base_dir, tuning_mode, base_name)

                gen_dir = os.path.join(this_dir, "generations_model", "eval")
                img_dir = os.path.join(this_dir, 'generations_score')
                log_path = os.path.join(this_dir, 'log.out')

                scratch_dir = os.path.join(scratch_base, f"{str(uuid.uuid4())}")
                command = (
                    f"python -m gpt2stuff.eval.eval_generations "
                    f"--task eval_trajectory --gen_dir {gen_dir} --img_dir {img_dir} "
                    f"--scratch_dir {scratch_dir} &"
                )
                command += '\n'
                commands += command

    script_path = os.path.join('.', 'gpt2stuff', 'scripts', f'gather_scaling_080621.sh')
    with open(script_path, 'w') as f:
        f.write(commands)


if __name__ == "__main__":
    fire.Fire(main)
