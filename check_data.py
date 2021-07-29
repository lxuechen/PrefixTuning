import fire
import numpy as np

import transformers


def main(
    task_mode="triples",
    percentiles=(50, 75, 90, 95, 99),
):
    file_path = {
        "triples": "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt"
    }[task_mode]

    with open(file_path, encoding="utf-8") as f:
        lines = [
            line.split('||')
            for line in f.read().splitlines() if (
                len(line) > 0 and not line.isspace() and len(line.split('||')) == 2
            )
        ]

    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    src_lines, tgt_lines = list(zip(*lines))
    src_lines = list(src_lines)
    tgt_lines = list(tgt_lines)

    lengths = []
    for src, tgt in zip(src_lines, tgt_lines):
        sent = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
        tokenized_sent = tokenizer.tokenize(sent)
        lengths.append(len(tokenized_sent))
    lengths = np.array(lengths)

    for percentile in percentiles:
        l = np.percentile(lengths, percentile)
        print(f'length {l} at percentile {percentile}')
    print(f'avg length: {np.mean(lengths)}')


if __name__ == "__main__":
    # Check lengths of various percentiles.
    # python check_data.py
    fire.Fire(main)
