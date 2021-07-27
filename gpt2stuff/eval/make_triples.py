import json
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
        references_no_space = []

        for example in file:
            if len(example['annotations']) == 0:  # Ignore empty.
                continue

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
            this_references_no_space = []
            for sent in example['annotations']:
                text = sent['text'].strip()
                this_references.append(text + '\n')
                this_references_no_space.append(text)
            this_references.append('\n')

            references.extend(this_references)
            references_no_space.append(this_references_no_space)

        print(f'split {split}, num prompts {len(prompts)}')

        prompts_path = os.path.join(base_dir, f'prompts_{split}.txt')
        with open(prompts_path, 'w') as f:
            f.writelines(prompts)

        references_path = os.path.join(base_dir, f'clean_references_{split}.txt')
        with open(references_path, 'w') as f:
            f.writelines(references)

        json_references_path = os.path.join(base_dir, f'json_clean_references_{split}.json')
        with open(json_references_path, 'w') as f:
            json.dump(references_no_space, f)


def main(task="make_triples"):
    if task == "make_triples":
        _make_triples()


if __name__ == "__main__":
    # python -m gpt2stuff.eval.make_triples
    fire.Fire(main)
