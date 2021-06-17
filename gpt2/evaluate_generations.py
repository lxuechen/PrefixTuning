import fire

import collections
import os


def clean(
    test_file_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt",
    out_file_path=None,
):
    """Clean the test file and create curated references.

    The references should be grouped.
    """
    with open(test_file_path, 'r') as f:
        lines = f.readlines()

    # TODO: The full-stop sign a bit awkward!
    src2tgt = collections.OrderedDict()
    for line_idx, line in enumerate(lines):
        src, tgt = line.split('||')
        if src not in src2tgt:
            src2tgt[src] = [tgt]
        else:
            src2tgt[src].append(tgt)

    del src, tgt

    if out_file_path is None:
        # TODO: This name is arbitrary!
        out_file_path = os.path.join(os.path.dirname(test_file_path), 'clean_reference_test.txt')

    # Write references to file.
    with open(out_file_path, 'w') as f:
        for src in src2tgt:
            for tgt in src2tgt[src]:
                f.write(tgt)
            f.write('\n')


# @formatter:off
def evaluate(
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0616/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_00100000/0/generations/eval/global_step_00001000.txt",
    ref_dir="",
):
    pass
# @formatter:on


def main(task="clean"):
    if task == "clean":
        clean()
    elif task == "evaluate":
        evaluate()
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
