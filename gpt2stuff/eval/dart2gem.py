"""Convert the dart reference and generation files to GEM format.

python -m gpt2stuff.eval.dart2gem
"""

import os

import fire

from lxuechen_utils import utils


def convert_ref(
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-test.json",
    out_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/gem-dart-v1.1.1-full-test.json"
):
    """Convert the references to the ref file format needed for GEM-metrics."""
    ref_dict = dict(language="en", values=[])
    data = utils.jload(ref_path)

    for i, example in enumerate(data):
        if len(example['annotations']) == 0:  # Only prompt but with no annotation!
            continue

        targets = [annotation["text"] for annotation in example['annotations']]
        ref_dict["values"].append(
            {
                "target": targets,
                "gem_id": i  # Still the sequential id.
            }
        )
    utils.jdump(ref_dict, out_path)


def convert_gen(
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-test.json",
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0720"
            "/model_name_gpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/generations_model/eval/",
):
    """Convert the generations to be the out file format needed GEM-metrics.

    Outputs to `gen_dir/../../gem_generations_model/eval`
    """
    data = utils.jload(ref_path)

    parpardir = os.path.abspath(os.path.join(gen_dir, os.pardir, os.pardir))
    new_gen_dir = os.path.join(parpardir, 'gem_generations_model', 'eval')
    os.makedirs(new_gen_dir, exist_ok=True)

    for gen_path in utils.listfiles(gen_dir):
        base_name = os.path.basename(gen_path)
        new_gen_path = os.path.join(new_gen_dir, base_name)

        with open(gen_path, 'r') as f:
            generations = f.readlines()

        gen_dict = dict(language="en", task="table2text", values=[])
        counter = 0  # Index the generation file.
        for idx, example in enumerate(data):
            if len(example['annotations']) == 0:  # Only prompt but with no annotation!
                continue

            gen_dict["values"].append(
                {
                    "generated": generations[counter],
                    "gem_id": idx,
                }
            )
            counter += 1
        assert counter == len(generations)
        utils.jdump(gen_dict, new_gen_path)


def main(task="convert_ref"):
    if task == "convert_ref":
        convert_ref()
    elif task == "convert_gen":
        convert_gen()


if __name__ == "__main__":
    fire.Fire(main)
