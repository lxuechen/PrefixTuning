"""Test out the basic exposure attacks."""
# python -m attacks.main
from .secret_sharer.generate_secrets import SecretConfig, generate_secrets_and_references

# 4 patterns, 4 vocabs.
patterns = [
    '{}' * 5,
    'Info ' + '{}' * 5,
    '{}{}-{}{}-{}',
    '{}' * 10,
]
vocabs = [list('ABCDEFGHIJ')] * len(patterns)

# A list of the possible number of repetitions, e.g. [2, 3] means that canaries appear either 2 or 3 times.
num_repetitions = [1]
num_secrets_for_repetitions = [3] * len(num_repetitions)
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
