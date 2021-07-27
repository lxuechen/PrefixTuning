# Dart test set is stupid as there are some prompts without references.
# This file cleans the generation files with full prompts to only include subset with references.

import re

import fire

from lxuechen_utils import utils


def sanitize_dir(
    data_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-test.json",
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0720"
            "/model_name_gpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/generations_model/eval/",
):
    files = []
    for file in utils.listfiles(gen_dir):
        search = re.search(".*global_step_(.*).txt", file)
        if search:
            files.append(file)
    print(f'sanitizing {len(files)} files')

    for file in files:
        sanitize_file(
            data_path=data_path,
            gen_path=file,
        )


def sanitize_file(
    data_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-test.json",
    gen_path="/nlp/scr/lxuechen/prefixtune/date_0720"
             "/model_name_gpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/generations_model/eval/global_step_00005900.txt",
    out_path=None,
):
    data = utils.jload(data_path)
    with open(gen_path, 'r') as f:
        generations = f.readlines()

    if len(data) == len(generations):  # Only run this if the number of generations is equal to the raw data size.
        generations_with_refs = []
        for i, (example, generation) in enumerate(zip(data, generations)):
            if len(example['annotations']) == 0:  # Only prompt but with no annotation!
                continue
            generations_with_refs.append(generation)

        if out_path is None:
            out_path = gen_path
        with open(out_path, 'w') as f:
            f.writelines(generations_with_refs)


def main(
    task="sanitize_dir",
    **kwargs,
):
    if task == "sanitize_dir":
        sanitize_dir(**kwargs)
    elif task == "sanitize_file":
        sanitize_file(**kwargs)


if __name__ == '__main__':
    # python -m gpt2stuff.eval.sanitize_dart
    fire.Fire(main)
