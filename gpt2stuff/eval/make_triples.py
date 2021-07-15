import os

import fire
import transformers

from lxuechen_utils import utils

base_dir = "/nlp/scr/lxuechen/data/prefix-tuning/data/dart"

split2file = {
    'train': f"{base_dir}/dart-v1.1.1-full-train.json",
    'valid': f"{base_dir}/dart-v1.1.1-full-dev.json",
    'test': f"{base_dir}/dart-v1.1.1-full-test.json",
}


def _create_default_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def _make_triples():
    tokenizer = _create_default_tokenizer()

    for split in ('train', 'valid', 'test'):
        file_path = split2file[split]
        file = utils.jload(file_path)

        prompts = []
        references = []

        for example in file:
            this_prompt = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                rela = rela.lower()
                if i > 0:
                    this_prompt += ' | '
                this_prompt += '{} : {} : {}'.format(subj, rela, obj)
            this_prompt += f" {tokenizer.bos_token} \n"
            prompts.append(this_prompt)

            this_references = []
            for sent in example['annotations']:
                this_references.append(sent['text'] + '\n')
            this_references.append('\n')
            references.extend(this_references)

        print(f'split {split}, num prompts {len(prompts)}')

        prompts_path = os.path.join(base_dir, f'prompts_{split}.txt')
        with open(prompts_path, 'w') as f:
            f.writelines(prompts)

        references_path = os.path.join(base_dir, f'clean_references_{split}.txt')
        with open(references_path, 'w') as f:
            f.writelines(references)


def main(task="make_triples"):
    if task == "make_triples":
        _make_triples()


if __name__ == "__main__":
    # python -m gpt2stuff.eval.make_triples
    fire.Fire(main)
