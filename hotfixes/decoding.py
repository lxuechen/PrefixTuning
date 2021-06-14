# Test out how decoding works.
# python decoding.py
import torch

import transformers

model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='gpt2-medium')
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path='gpt2-medium')

# Check out how beam-search works.
input_ids = torch.tensor([[4, 6, 9, 10]])
# TODO: How is max length selected?
# TODO: What are the bad_words?
bad_words = tokenizer.decode([628, 198])
print('bad words')
print(f"{len(bad_words)}")
print(f"{[repr(i) for i in bad_words]}")
print('')

# TODO: past_key_values?
model.eval()
outputs = model.generate(
    input_ids=input_ids,
    min_length=5,
    max_length=20,
    top_k=0,
    top_p=0.9,  # Only filter with top_p.
    do_sample=False,
    num_beams=5,
    bad_words_ids=[[628], [198]],
    num_return_sequences=1,
)
print(input_ids)
print(outputs)
# TODO: It seems outputs repeats inputs!
