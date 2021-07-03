from itertools import chain

import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel


def test_chatbot_example():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    SPECIAL_TOKENS = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "additional_special_tokens": ("<speaker1>", "<speaker2>")
    }
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print('special_tokens_map', tokenizer.special_tokens_map)

    # Let's define our contexts and special tokens
    persona = [["i", "like", "playing", "football", "."], ["i", "am", "from", "NYC", "."]]
    history = [["hello", "how", "are", "you", "?"], ["i", "am", "fine", "thanks", "."]]
    reply = ["great", "to", "hear"]
    bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

    def build_inputs(persona, history, reply):
        # Build our sequence by adding delimiters and concatenating
        sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
        sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        # Build our word, segments and position inputs from the sequence
        words = list(chain(*sequence))  # word tokens
        segments = [speaker2 if i % 2 else speaker1  # segment tokens
                    for i, s in enumerate(sequence) for _ in s]
        position = list(range(len(words)))  # position tokens
        return words, segments, position, sequence

    words, segments, position, sequence = build_inputs(persona, history, reply)

    # Tokenize words and segments embeddings:
    words = tokenizer.convert_tokens_to_ids(words)
    segments = tokenizer.convert_tokens_to_ids(segments)

    # lxuechen: Here starts the third code segment.

    # Let's add a distractor to our previously defined persona, history and reply
    distractor = ["sorry", "to", "hear", "that"]

    # Build & tokenize inputs ending with our distractor like we did with the gold reply
    words_distractor, segments_distractor, _, _ = build_inputs(persona, history, distractor)
    words_distractor = tokenizer.convert_tokens_to_ids(words_distractor)
    segments_distractor = tokenizer.convert_tokens_to_ids(segments_distractor)
    # lxuechen: Check these should not be 0 (0 is for unknown token, or '<unk>').
    print('segments_distractor', segments_distractor)

    # Prepare our language modeling targets: keep only the reply segment, -1 on the rest
    lm_targets = (([-1] * sum(len(s) for s in sequence[:-1])) +
                  [-1] + tokenizer.convert_tokens_to_ids(sequence[-1][1:]))
    lm_distractor = [-1] * len(words_distractor)

    # Store the position of the last tokens for the next-sentence prediction loss
    last_token = len(words) - 1
    last_token_distractor = len(words_distractor) - 1

    # Now we can pad reply and distractor inputs and targets to the same length
    padding_length = max(len(words), len(words_distractor))

    def pad(x, padding):
        return x + [padding] * (padding_length - len(x))

    (words, words_distractor, segments, segments_distractor) = [
        pad(x, tokenizer.convert_tokens_to_ids('<pad>'))
        for x in (words, words_distractor, segments, segments_distractor)
    ]

    # Need -1 to avoid computing the loss!
    (lm_targets, lm_distractor) = [pad(x, -1) for x in (lm_targets, lm_distractor)]

    # And gather reply and distractor inputs to build the input tensors: words tokens
    input_ids = torch.tensor([[words, words_distractor]], dtype=torch.long)

    # segment tokens
    # lxuechen: Check these aren't 0. They can't be to have things work correctly.
    token_type_ids = torch.tensor([[segments, segments_distractor]], dtype=torch.long)

    # Positions tokens can be automatically created by the model as (0, 1, ..., N)
    # Last tokens location
    mc_token_ids = torch.tensor([[last_token, last_token_distractor]], dtype=torch.long)
    # Language modeling labels
    lm_labels = torch.tensor([[lm_targets, lm_distractor]], dtype=torch.long)
    # Next-sentence prediction labels
    mc_labels = torch.tensor([0], dtype=torch.long)  # Gold reply is 1st (index 0)

    # TODO: Why should mc_labels have the current format?


@torch.no_grad()
def test_double_heads_model():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')

    choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
    input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
    mc_token_ids = torch.tensor([input_ids.size(-1) - 1, input_ids.size(-1) - 1]).unsqueeze(0)  # Batch size 1

    outputs = model(input_ids, mc_token_ids=mc_token_ids, return_dict=True)
    lm_logits = outputs.logits
    mc_logits = outputs.mc_logits

    print(lm_logits.size())
    print(mc_logits.size())

    print('loss', outputs.loss)
    print('mc loss', outputs.mc_loss)


if __name__ == '__main__':
    print('test 1')
    test_chatbot_example()

    print('test 2')
    test_double_heads_model()
