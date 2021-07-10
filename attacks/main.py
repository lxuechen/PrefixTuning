"""Test out the basic exposure attacks."""
import os
import pickle

from .secret_sharer.generate_secrets import SecretConfig, generate_secrets_and_references


def main():
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


if __name__ == "__main__":
    # python -m attacks.main
    main()
