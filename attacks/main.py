"""Test out the basic exposure attacks.

`_create_canaries` right now creates the "canarized" dataset for e2e.

To run:
python -m attacks.main
"""
import os
import pickle
import json

import fire
import numpy as np
import transformers

from .secret_sharer.generate_secrets import SecretConfig, generate_secrets_and_references


def _play():
    # 4 patterns matched with 4 vocabs.
    # Pattern determines the length of the sentence.
    # e2e pattern: "name : {} | Type : {} | area : {} || {} {} {} {} ."
    patterns = [
        '{}' * 5,
        'Info ' + '{}' * 5,
        '{}{}-{}{}-{}',
        '{}' * 10,
    ]
    vocabs = [list('ABCDEFGHIJ')] * len(patterns)

    # A list of the possible number of repetitions, e.g. [2, 3] means that canaries appear either 2 or 3 times.
    num_repetitions = [1]
    num_secrets_for_repetitions = [10] * len(num_repetitions)
    num_references = 65536
    secret_configs = [
        SecretConfig(
            vocab=vocab,
            pattern=pattern,
            num_repetitions=num_repetitions,
            num_secrets_for_repetitions=num_secrets_for_repetitions,
            num_references=num_references
        )
        for vocab, pattern in zip(vocabs, patterns)
    ]
    secrets = generate_secrets_and_references(secret_configs)

    out_path = os.path.join('.', 'attacks', 'test.json')
    with open(out_path, 'wb') as f:
        pickle.dump(secrets, f)

    with open(out_path, 'rb') as g:
        s2 = pickle.load(g)

    # Test both loading and dumping.
    print(secrets[0].secrets)
    print(s2[0].secrets)


def _create_canaries(num_references=2 ** 15, num_repetitions=(1,), num_secrets_for_repetitions=None, vocab_size=10):
    np.random.seed(13)

    if num_secrets_for_repetitions is None:
        num_secrets_for_repetitions = [20] * len(num_repetitions)

    tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2')

    # Only include whole words!
    vocab_ids = np.random.permutation(len(tokenizer))[:vocab_size]
    vocab = tokenizer.convert_ids_to_tokens(vocab_ids, skip_special_tokens=False)
    vocab = [word[len('Ġ'):] if word.startswith('Ġ') else word for word in vocab]
    print('vocab')
    print(vocab)

    # TODO: For some reason, adding newline for pattern just doesn't work.
    pattern = "name : {} | Type : {} | area : {} || {} {} {} {} {} ."
    secret_configs = [
        SecretConfig(
            vocab=vocab,
            pattern=pattern,
            num_repetitions=num_repetitions,
            num_secrets_for_repetitions=num_secrets_for_repetitions,
            num_references=num_references
        )
    ]
    secrets, = generate_secrets_and_references(secret_configs)

    # Load clean data, insert canaries, and store new data as well as the secrets file (it has all the references)!
    with open("/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt", 'r') as f:
        data = f.readlines()

    secret_data = []
    for reps, lines in secrets.secrets.items():
        for line in lines:
            secret_data += [line + '\n'] * reps
    data += secret_data

    src_dir = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/"
    tgt_dir = f"/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data-{num_repetitions[0]}/"
    print(tgt_dir)

    os.system(f"cp -r {src_dir} {tgt_dir}")
    print('copying dir')

    with open(f"{tgt_dir}/src1_train.txt", 'w') as f:
        f.writelines(data)
    del f

    with open(f"{tgt_dir}/ss_refs.txt", 'w') as f:
        refs = [line + '\n' for line in secrets.references]
        print(f'total refs: {len(refs)}')
        f.writelines(refs)
    del f

    with open(f"{tgt_dir}/ss_secs.txt", 'w') as f:
        secret_data_no_dup = []
        for reps, lines in secrets.secrets.items():
            for line in lines:
                secret_data_no_dup += [line + '\n']
        print(f'total secs (no dups): {len(secret_data_no_dup)}')
        f.writelines(secret_data_no_dup)
    del f

    info = {
        "num_references": num_references,
        "num_repetitions": num_repetitions,
        "num_secrets_for_repetitions": num_secrets_for_repetitions,
        "vocab_size": vocab_size,
    }
    with open(f"{tgt_dir}/info.txt") as f:
        json.dump(info, f, indent=4)


def main(task="create_canaries"):
    if task == "create_canaries":
        for num_repetitions in (
            (1,), (5,), (10,), (100,), (500,)
        ):
            _create_canaries(num_repetitions=num_repetitions)


def _sp():
    with open('shakespeare.txt', 'rb') as f:
        text = f.read().decode(encoding='utf-8')
        print(sorted(set(text)))


if __name__ == "__main__":
    fire.Fire(main)
