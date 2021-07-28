import flaxmodels as fm
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)

# Initialize tokenizer
tokenizer = fm.gpt2.get_tokenizer()

# Encode start sequence
generated = tokenizer.encode('The Manhattan bridge')

context = jnp.array([generated])
past = None

# Initialize model
# Models to choose from ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
model = fm.gpt2.GPT2LMHeadModel(pretrained='gpt2')
params = model.init(key, input_ids=context, past_key_values=past)

for i in range(20):
    # Predict next token in sequence
    output = model.apply(params, input_ids=context, past_key_values=past, use_cache=True)
    token = jnp.argmax(output['logits'][..., -1, :])
    context = jnp.expand_dims(token, axis=0)
    # Add token to sequence
    generated += [token]
    # Update past keys and values
    past = output['past_key_values']

# Decode sequence of tokens
sequence = tokenizer.decode(generated)
print(sequence)
