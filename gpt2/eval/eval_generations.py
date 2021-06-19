import collections
from datetime import date
import logging
import os
import shutil
import uuid

import fire
import transformers

from lxuechen_utils import utils


def _create_default_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def _extract_mapping(file_path, tokenizer=None):
    """Get the prompt (src) to reference generation (tgt) mapping."""
    if tokenizer is None:
        tokenizer = _create_default_tokenizer()

    with open(file_path, 'r') as f:
        lines = f.readlines()
    print(len(lines))

    # TODO: Punctuation a bit awkward!
    src2tgt = collections.OrderedDict()
    for line_idx, line in enumerate(lines):
        src, tgt = line.strip().split('||')
        src = f"{src} {tokenizer.bos_token}"
        if src not in src2tgt:
            src2tgt[src] = [tgt]
        else:
            src2tgt[src].append(tgt)
        del src, tgt

    return src2tgt


def extract_references(
    file_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt",
    out_file_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/clean_references_test.txt",
    tokenizer=None,
    **_,
):
    """Clean the test file and create curated references.

    The references should be grouped.
    """
    src2tgt = _extract_mapping(file_path, tokenizer=tokenizer)

    with open(out_file_path, 'w') as f:
        for src in src2tgt:
            for tgt in src2tgt[src]:
                if not tgt.endswith('\n'):
                    tgt = tgt + "\n"
                f.write(tgt)
            f.write('\n')
    logging.warning(f"Number of prompts for generation: {len(src2tgt)}")


def extract_prompts(
    file_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt",
    out_file_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_test.txt",
    tokenizer=None,
    **_,
):
    """Get all the prompts and deduplicate."""
    src2tgt = _extract_mapping(file_path, tokenizer=tokenizer)

    with open(out_file_path, 'w') as f:
        prompts = src2tgt.keys()
        giant_chunk = [line + '\n' for line in prompts]
        f.writelines(giant_chunk)  # TODO: You might need to strip '\n' eventually.
    logging.warning(f"Number of prompts for generation: {len(prompts)}")


def eval(
    # @formatter:off
    # Private.
    gen_path="/nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_00100000_train_batch_size_00000100/0/generations/eval/global_step_00002100.txt",
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/clean_references_test.txt",

    # Clone the e2e-metrics repo to this dir if you haven't already: https://github.com/lxuechen/e2e-metrics
    e2e_dir="/sailhome/lxuechen/software/e2e-metrics",
    # @formatter:on
):
    """Evaluate a file of generate sentences against references."""
    os.system(f'cd {e2e_dir}; ./measure_scores.py {ref_path} {gen_path} ; cd -')


def eval_trajectory(
    # @formatter:off
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/clean_references_test.txt",
    # Non-private.
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0619/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010/0/generations/eval/",
    e2e_dir="/sailhome/lxuechen/software/e2e-metrics",
    scratch_dir="/nlp/scr/lxuechen/scratch/tmp",  # Mess around here.
    global_steps=tuple(range(1000, 42001, 1000)) + (42025,),

    img_dir="/nlp/scr/lxuechen/plots/distilgpt2-e2e-nonprivate",
    # @formatter:on
):
    """Evaluate various scores and plot trajectory."""
    # Check the files exist.
    for global_step in global_steps:
        gen_path = os.path.join(gen_dir, f"global_step_{global_step:08d}.txt")
        assert os.path.exists(gen_path), f"Failed to find path {gen_path}"
        del gen_path

    logging.warning(f"eval_trajectory for gen_dir {gen_dir}")

    os.makedirs(scratch_dir, exist_ok=True)
    scores = []
    for global_step in global_steps:
        gen_path = os.path.join(gen_dir, f"global_step_{global_step:08d}.txt")
        out_path = os.path.join(scratch_dir, f'global_step_{global_step:08d}.json')
        os.system(f'cd {e2e_dir}; ./measure_scores.py {ref_path} {gen_path} --out_path {out_path}; cd -')

        score = utils.jload(out_path)
        scores.append(score)
        del score
    shutil.rmtree(scratch_dir)

    metrics = scores[0].keys()
    for metric in metrics:
        x = global_steps
        y = [score[metric] for score in scores]
        img_path = os.path.join(img_dir, f"{metric}.png")
        utils.plot(
            plots=({'x': x, 'y': y, 'label': metric},),
            options={"xlabel": "steps", "ylabel": "metric"},
            img_path=img_path,
        )


def gen2ref(
    # @formatter:off
    gen_path="/nlp/scr/lxuechen/prefixtune/date_0619/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_00100000_train_batch_size_00000100_mid_dim_00000512_preseqlen_00000010/0/generations/eval/global_step_00002100.txt",
    file_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt",
    tokenizer=None,

    uid_max_len=8,
    img_dir="/nlp/scr/lxuechen/plots/distilgpt2-e2e-nonprivate",
    # @formatter:on
):
    # TODO: This is dumb; should just use the prompt as the key.
    if tokenizer is None:
        tokenizer = _create_default_tokenizer()

    with open(file_path, 'r') as f:
        true_lines = f.readlines()

    with open(gen_path, 'r') as g:
        gen_lines = g.readlines()

    # Add uids, since you might get repeated generations.
    gen_lines = [f"uuid4={str(uuid.uuid4())[:uid_max_len]} || " + line.strip() for line in gen_lines]

    gen2ref_map = dict()
    gen_idx = -1

    src2tgt = collections.OrderedDict()
    for line_idx, true_line in enumerate(true_lines):
        src, tgt = true_line.strip().split('||')
        src = f"{src} {tokenizer.bos_token}"
        if src not in src2tgt:
            src2tgt[src] = [tgt]

            gen_idx += 1
            gen2ref_map[gen_lines[gen_idx]] = [tgt]
        else:
            src2tgt[src].append(tgt)

            gen2ref_map[gen_lines[gen_idx]].append(tgt)
        del src, tgt

    today = date.today().strftime("%m%d%y")
    out_path = os.path.join(img_dir, f'distilgpt2-private-head2head-comparison-{today}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    utils.jdump(gen2ref_map, out_path)

    return src2tgt, gen2ref_map


def main(task="clean", **kwargs):
    # python -m gpt2.eval.eval_generations --task clean
    if task == "clean":
        for split in ('valid', 'test', 'train'):
            file_path = f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_{split}.txt"

            out_file_path = f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/clean_references_{split}.txt"
            extract_references(file_path=file_path, out_file_path=out_file_path, **kwargs)

            out_file_path = f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_{split}.txt"
            extract_prompts(file_path=file_path, out_file_path=out_file_path, **kwargs)

    # python -m gpt2.eval.eval_generations --task eval
    elif task == "eval":
        eval(**kwargs)

    elif task == "eval_nonprivate":
        # @formatter:off
        gen_path = "/nlp/scr/lxuechen/prefixtune/date_0619/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010/0/generations/eval/global_step_00042000.txt"
        eval(gen_path=gen_path)
        # @formatter:on

    elif task == "eval_trajectory":
        eval_trajectory(**kwargs)

    elif task == "eval_best_private_trajectory":
        # python -m gpt2.eval.eval_generations --task eval_best_private_trajectory

        # Private.
        # @formatter:off

        # Small batch size and small learning rate.

        # gen_dir = "/nlp/scr/lxuechen/prefixtune/date_0620/" \
        #           "model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_00100000_train_batch_size_00000100_mid_dim_00000512_preseqlen_00000010/0/generations/eval/"
        # global_steps = tuple(range(500, 15001, 500))
        # img_dir = "/nlp/scr/lxuechen/plots/distilgpt2-e2e-private"

        # Large batch size and large learning rate.
        gen_dir = "/nlp/scr/lxuechen/prefixtune/date_0620/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_01000000_train_batch_size_00000500_mid_dim_00000512_preseqlen_00000010/0/generations/eval/"
        global_steps = tuple(range(500, 7001, 500))
        img_dir = "/nlp/scr/lxuechen/plots/distilgpt2-e2e-private-large-bs-large-lr"

        # @formatter:on
        eval_trajectory(
            gen_dir=gen_dir,
            global_steps=global_steps,
            img_dir=img_dir,
        )

    # python -m gpt2.eval.eval_generations --task gen2ref
    elif task == "gen2ref":
        gen2ref(**kwargs)

    # Best private generation for today!
    # python -m gpt2.eval.eval_generations --task gen2ref_061821
    elif task == "gen2ref_061821":
        gen2ref(
            gen_path="/nlp/scr/lxuechen/prefixtune/date_0620/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_00100000_train_batch_size_00000100_mid_dim_00000512_preseqlen_00000010/0/generations/eval/global_step_00013500.txt",
            img_dir="/nlp/scr/lxuechen/plots/distilgpt2-e2e-private",
        )
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)

# This is the best private generation so far with distilgpt2.
# @formatter:off
# /nlp/scr/lxuechen/prefixtune/date_0620/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_00100000_train_batch_size_00000100_mid_dim_00000512_preseqlen_00000010/0/generations/eval/global_step_00007000.txt
# @formatter:on
