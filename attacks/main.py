"""Test out the basic exposure attacks."""
import os
import pickle

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
    if num_secrets_for_repetitions is None:
        num_secrets_for_repetitions = [40] * len(num_repetitions)

    tokenizer = transformers.AutoTokenizer.from_pretrained('distilgpt2')

    # Only include whole words!
    vocab_ids = np.random.permutation(len(tokenizer))[:vocab_size]
    vocab = tokenizer.convert_ids_to_tokens(vocab_ids, skip_special_tokens=False)
    vocab = [word[len('Ġ'):] if word.startswith('Ġ') else word for word in vocab]

    pattern = "name : {} | Type : {} | area : {} || {} {} {} {} ."
    secret_configs = [
        SecretConfig(
            vocab=vocab,
            pattern=pattern,
            num_repetitions=num_repetitions,
            num_secrets_for_repetitions=num_secrets_for_repetitions,
            num_references=num_references
        )
    ]
    secrets = generate_secrets_and_references(secret_configs)
    print(len(secrets))
    print(secrets[0].secrets)


def main(task="create_canaries"):
    if task == "create_canaries":
        for num_repetitions in (
            (1,), (10,), (100,)
        ):
            _create_canaries(num_repetitions=num_repetitions)


def _sp():
    with open('shakespeare.txt', 'rb') as f:
        text = f.read().decode(encoding='utf-8')
        print(sorted(set(text)))


if __name__ == "__main__":
    # python -m attacks.main
    np.random.seed(13)
    fire.Fire(main)
