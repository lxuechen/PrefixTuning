import sys
from typing import Optional
import tqdm


def generate(
    loader, model, tokenizer,
    max_length=20,
    min_length=5,
    top_k=0,
    top_p=0.9,  # Only filter with top_p.
    repetition_penalty=1,
    do_sample=False,
    num_beams=5,
    bad_words_ids=None,
    dummy_token_id=-100,  # Used as mask.
    num_return_sequences=1,
    max_generations=sys.maxsize,
    device=None,
):
    assert not model.training, "Generation must be when `model` is in eval mode."

    # These are linebreaks; generating these will mess up the evaluation, since those files assume one example per-line.
    if bad_words_ids is None:
        bad_words_ids = [[628], [198]]

    generations = []
    for batch_idx, batch in tqdm.tqdm(enumerate(loader), desc="generation"):
        batch_input_ids, batch_labels = batch["input_ids"], batch["labels"]
        # e.g., inputs_ids may be [[95, 123, 32], [198, 19, 120]], and
        # labels may be [[-100, 123, 32], [-100, -100, 120]

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            # Find the first non- -100 position. Note there are trailing -100s.
            non_prompt_positions, = (labels != dummy_token_id).nonzero(as_tuple=True)
            first_non_prompt_position = non_prompt_positions[0].item()
            prompt_len = first_non_prompt_position
            input_ids = input_ids[:prompt_len]

            output_ids = model.generate(
                input_ids=input_ids[None, ...].to(device),
                max_length=max_length + prompt_len,
                min_length=min_length,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                pad_token_id=tokenizer.eos_token_id,  # Stop the stupid logging...
            )
            output_ids = output_ids.squeeze(dim=0)  # Throw away batch dimension.

            whole_str: str = tokenizer.decode(output_ids, clean_up_tokenization_spaces=True)
            prompt_str: str = tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)

            eos_position: Optional[int] = whole_str.find(tokenizer.eos_token)
            if eos_position == -1:  # Didn't generate eos_token; that's okay -- just skip!
                eos_position = None
            output_str: str = whole_str[len(prompt_str):eos_position]
            output_str = output_str.strip()

            generations.append(output_str)

        if len(generations) >= max_generations:
            break

    return generations
