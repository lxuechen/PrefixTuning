import collections
import logging

import fire
import transformers


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
        f.writelines('\n'.join(prompts))  # TODO: You might need to strip '\n' eventually.
    logging.warning(f"Number of prompts for generation: {len(prompts)}")


def eval(
    # @formatter:off
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0616/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_01000000/0/generations/eval/global_step_00001400.txt",
    ref_dir="",
    # @formatter:on
):
    pass


def main(task="clean", **kwargs):
    # python -m gpt2.evaluate_generations --task clean
    if task == "clean":
        for split in ('valid', 'test', 'train'):
            file_path = f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_{split}.txt"

            out_file_path = f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/clean_references_{split}.txt"
            extract_references(file_path=file_path, out_file_path=out_file_path, **kwargs)

            out_file_path = f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_{split}.txt"
            extract_prompts(file_path=file_path, out_file_path=out_file_path, **kwargs)
    elif task == "eval":
        eval(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
