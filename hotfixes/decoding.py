# Test out how decoding works.
# python decoding.py
import torch

import transformers

model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='gpt2-medium')
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path='gpt2-medium')

# Check out how beam-search works.
input_text = "James bond is working at "
input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
# TODO: How is max length selected?
bad_words = tokenizer.decode([628, 198])
print('bad words')
print(f"{len(bad_words)}")
print(f"{[repr(i) for i in bad_words]}")
print('')

# TODO: Check this is the correct setup.
# TODO: Ensure things are correct when we have prompt.
model.eval()
outputs = model.generate(
    input_ids=input_ids,
    min_length=5,
    max_length=20,
    top_k=0,
    top_p=0.9,  # Only filter with top_p.
    do_sample=False,
    num_beams=5,
    # These are linebreaks; generating these will mess up the evaluation, since those files assume one example per-line.
    bad_words_ids=[[628], [198]],
    num_return_sequences=1,
)
print(input_ids)
print(outputs)
print(outputs.size())

text_prefix = tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)
text_output = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)

print('eos: ')
print(f'{repr(tokenizer.eos_token)}')

print(f'{repr(text_prefix)}')
print(f'{repr(text_output)}')
print(text_output[len(text_prefix):])

idx = text_output.find(tokenizer.eos_token)
if idx > 0:
    print(text_output[len(text_prefix):idx])
else:
    print(text_output[len(text_prefix)])

# TODO: It seems outputs repeats inputs!

import torch.nn.functional as F

logits = torch.randn(2, 100)
labels = torch.tensor([-100, 0])
loss = F.cross_entropy(logits, labels)
print(loss)
