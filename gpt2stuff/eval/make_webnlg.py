import os

import fire
import transformers

from lxuechen_utils import utils

base_dir = "/nlp/scr/lxuechen/data/prefix-tuning/data/webnlg_challenge_2017"

split2file = {
    'train': f"{base_dir}/train.json",
    'valid': f"{base_dir}/dev.json",
    'test': f"{base_dir}/test.json",
}


def _create_default_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def _make_webnlg():
    tokenizer = _create_default_tokenizer()

    for split in ('train', 'valid', 'test'):
        file_path = split2file[split]
        file = utils.jload(file_path)

        prompts = []
        references = []

        for i, example in enumerate(file['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            this_prompt = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                this_prompt += ' | '
                this_prompt += '{} : {} : {}'.format(subj, rela, obj)
            this_prompt += f" {tokenizer.bos_token} \n"

            this_references = []
            for sent in sents:
                if sent["comment"] == 'good':
                    this_references.append(sent["lex"] + '\n')

            prompts.append(this_prompt)

            this_references.append('\n')
            references.extend(this_references)

        print(f'split {split}, num prompts {len(prompts)}')

        prompts_path = os.path.join(base_dir, f'prompts_{split}.txt')
        with open(prompts_path, 'w') as f:
            f.writelines(prompts)

        references_path = os.path.join(base_dir, f'clean_references_{split}.txt')
        with open(references_path, 'w') as f:
            f.writelines(references)


def main(task="make_webnlg"):
    if task == "make_webnlg":
        _make_webnlg()


if __name__ == "__main__":
    # python -m gpt2stuff.eval.make_webnlg
    fire.Fire(main)
