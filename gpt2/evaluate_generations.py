import collections
import os

import fire
import logging


def clean(
    test_file_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt",
    scratch_dir="/nlp/scr/lxuechen/scratch",
    out_file_path=None,
):
    """Clean the test file and create curated references.

    The references should be grouped.
    """
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    print(len(lines))

    # TODO: Punctuation a bit awkward!
    src2tgt = collections.OrderedDict()
    for line_idx, line in enumerate(lines):
        src, tgt = line.split('||')
        if src not in src2tgt:
            src2tgt[src] = [tgt]
        else:
            src2tgt[src].append(tgt)
    del src, tgt

    if out_file_path is None:
        out_file_path = os.path.join(scratch_dir, 'clean_reference_test.txt')

    # Write references to file.
    with open(out_file_path, 'w') as f:
        for src in src2tgt:
            for tgt in src2tgt[src]:
                f.write(tgt)
            f.write('\n')
    logging.warning(f"Number of prompts for generation: {len(src2tgt)}")


# You probably need to throw this away!
def eval_old(
    # @formatter:off
    gen_path="/nlp/scr/lxuechen/prefixtune/date_0616/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_01000000/0/generations/eval/global_step_00001400.txt",
    ref_path="/nlp/scr/lxuechen/scratch/clean_reference_test.txt",
    scratch_dir="/nlp/scr/lxuechen/scratch",

    # TODO: cd to the official evaluation dir and run the evaluation.
    eval_script_path="",
    # @formatter:on
):
    """The generations are repeated, since the same prompt is used many times.

    The same prompt may lead to many possible generations in the training split.
    I wasn't careful about checking the data preprocessing.
    """
    with open(gen_path, 'r') as f:
        lines = f.readlines()
    print(len(lines))

    # Deduplicate with specific ordering!
    deduplicated_lines = []
    last_line = None
    for line in lines:
        if line != last_line:
            deduplicated_lines.append(line)
            last_line = line
    print(f"Number of deduplicated lines: {len(deduplicated_lines)}")


# TODO: This evaluation should be with the new generation script that blocks off repeated prompts!
def eval(
    # @formatter:off
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0616/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_01000000/0/generations/eval/global_step_00001400.txt",
    ref_dir="",
    # @formatter:on
):
    pass


def main(task="clean", **kwargs):
    if task == "clean":
        clean(**kwargs)
    elif task == "eval_old":
        eval_old(**kwargs)
    elif task == "eval":
        eval(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
