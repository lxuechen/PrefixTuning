import json

import fire
import numpy as np

import transformers


def main(
    task_mode="data2text",
    percentiles=(50, 75, 90, 95, 99),
):
    file_path = {
        "data2text": "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt",
        'triples': "/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-train.json"
    }[task_mode]

    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    if task_mode == "data2text":
        with open(file_path, encoding="utf-8") as f:
            lines = [
                line.split('||')
                for line in f.read().splitlines() if (
                    len(line) > 0 and not line.isspace() and len(line.split('||')) == 2
                )
            ]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        sents = []
        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
            sents.append(sent)
    else:
        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        for example in lines_dict:
            rela_lst = []
            temp_triples = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                rela = rela.lower()
                rela_lst.append(rela)
                if i > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in example['annotations']:
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)

        sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = f' {src} {tokenizer.bos_token} {tgt} {tokenizer.eos_token} '
            sents.append(sent)

    lengths = np.array([len(tokenizer.tokenize(sent)) for sent in sents])
    for percentile in percentiles:
        l = np.percentile(lengths, percentile)
        print(f'length {l} at percentile {percentile}')
    print(f'avg length: {np.mean(lengths)}')


if __name__ == "__main__":
    # Check lengths of various percentiles.
    # python check_data.py
    # python check_data.py --task_mode "triples"
    fire.Fire(main)
