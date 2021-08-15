"""Quickly check what the models are like; how many parameters, how many layers?

python inspect_model.py
"""

import os

import transformers

home_dir = os.path.expanduser('~')
cache_dir = os.path.join(home_dir, '.cache', 'gpt-models')
os.makedirs(cache_dir, exist_ok=True)

model_cards = (
    "openai-gpt", "distilgpt2", 'gpt2', "gpt2-medium", "gpt2-large",
)

untied_params = dict()
params = dict()
embd_params = dict()

for model_card in model_cards:
    model = transformers.AutoModelWithLMHead.from_pretrained(
        model_card, cache_dir=cache_dir
    )
    num_params = sum(param.numel() for param in model.parameters())
    num_embd_params = model.get_input_embeddings().weight.numel()
    num_untied_params = num_params + num_embd_params

    print('---------------------')
    print(f'model: {model_card}')
    print(type(model.config))
    print(f'config: {model.config}')

    print(f'tied model params: {num_params / 1e6:.4f}')
    print(f'untied model params: {num_untied_params / 1e6:.4f}')
    print(f'embd params: {num_embd_params / 1e6:.4f}')

    params[model_card] = num_params
    untied_params[model_card] = num_untied_params
    embd_params[model_card] = num_embd_params

print('tied')
print(repr(params))

print('untied')
print(repr(untied_params))

print('embd')
print(repr(embd_params))
