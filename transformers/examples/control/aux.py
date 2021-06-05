import torch


def quick_generate(
    discri_labels, discri_labels_code, input_ids_prompt, prompt_text, model, gpt2, tokenizer,
    sample_size=10, sample_from_gpt=False,
    textlength=50, nolinebreak=True, stop_token='[EOS]'
):
    control_codes = []
    sst_codes = []
    prompt_codes = []
    for a in range(len(discri_labels)):
        control_code = discri_labels_code[a]
        control_codes += [control_code] * sample_size
        sst_codes += [a] * sample_size
        if not sample_from_gpt:
            prompt = model.get_prompt(control_code, gpt2)
            prompt = [x.expand(-1, sample_size, -1, -1, -1) for x in
                      prompt]  # (2, batch_size, num_heads, sequence_length, embed_size_per_head)
        else:
            prompt = None
        prompt_codes.append(prompt)

    if not sample_from_gpt:
        prompt_codes = list(zip(*prompt_codes))
        prompt_full = []
        for prompt_c in prompt_codes:
            prompt_c = torch.cat(prompt_c, dim=1)
            prompt_full.append(prompt_c)
    else:
        prompt_full = None

    full_results = gpt2.generate(
        input_ids=input_ids_prompt,
        emb_match=None,
        control_code=None,
        past_key_values=prompt_full,
        max_length=textlength,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=sample_size * len(discri_labels),
        bad_words_ids=[[628], [198]] if nolinebreak else None,
        use_cache=True
    )

    print(full_results)

    for generated_sequence_idx, generated_sequence in enumerate(full_results):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        if input_ids_prompt is not None:
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(input_ids_prompt[0], clean_up_tokenization_spaces=True)):]
            )
        else:
            total_sequence = (text)

        print(discri_labels[sst_codes[generated_sequence_idx]])
        print(total_sequence)

    return {'input_ids': full_results, 'control_code': control_codes, 'sst_codes': sst_codes}
