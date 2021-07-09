"""Test out the basic exposure attacks."""
# python -m attacks.main
from .secret_sharer.generate_secrets import SecretConfig, generate_secrets_and_references

# 4 patterns, 4 vocabs.
patterns = [
    '{}' * 5,
    'Info ' + '{}' * 5,
    '{}{}-{}{}-{}',
    '{}' * 10
]
vocabs = [list('ABCDEFGHIJ')] * 4

num_repetitions = [1, 10, 100]
num_secrets_for_repetitions = [20] * len(num_repetitions)
num_references = 65536
secret_configs = [
    SecretConfig(vocab, pattern, num_repetitions, num_secrets_for_repetitions, num_references)
    for vocab, pattern in zip(vocabs, patterns)
]
secrets = generate_secrets_and_references(secret_configs)
print(secrets)
