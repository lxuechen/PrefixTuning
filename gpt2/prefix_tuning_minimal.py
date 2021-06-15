"""Make the prefix-tuning model as minimal as possible."""
import torch
from torch import nn
from transformers import GPT2PreTrainedModel, GPT2LMHeadModel


class _View(nn.Module):
    def __init__(self, shape):
        super(_View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class PrefixTuningMinimal(GPT2PreTrainedModel):
    """A minimalistic implementation of the core components."""

    def __init__(self, config, model_args, gpt2=None):
        super(PrefixTuningMinimal, self).__init__(config=config)

        # Instantiate a GPT-2, and DON'T optimizer it!
        if gpt2 is None:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(
                model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir,
            ).requires_grad_(False)
        else:
            self.gpt2 = gpt2.requires_grad_(False)

        self.register_buffer('extra_prefix_ids', torch.arange(model_args.preseqlen))
        self.extra_prefix_net = nn.Sequential(
            nn.Embedding(model_args.preseqlen, config.n_embd),
            nn.Linear(config.n_embd, model_args.mid_dim),
            nn.Tanh(),
            nn.Linear(model_args.mid_dim, config.n_layer * 2 * config.n_embd),
            _View((-1, model_args.preseqlen, config.n_layer * 2, config.n_head, config.n_embd // config.n_head)),
            nn.Dropout(model_args.prefix_dropout),
        )

    def make_past_key_values(self, bsz=None):
        extra_prefix_ids = self.extra_prefix_ids[None, :].expand(bsz, -1)
        past_key_values = self.extra_prefix_net(extra_prefix_ids)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def state_dict(self):
        """Avoid storing GPT-2, since it's not even trained."""
        return self.prefix_net.state_dict()

    def load_state_dict(self, state_dict):
        """Avoid loading GPT-2, since it's not even trained."""
        self.prefix_net.load_state_dict(state_dict)

    @property
    def major_device(self):
        """Returns the device where the parameters are on."""
        return next(self.parameters()).device

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        past_key_values = self.make_past_key_values(bsz=input_ids.size(0))
        return self.gpt2(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def generate(self, input_ids, **kwargs):
        past_key_values = self.make_past_key_values(bsz=input_ids.size(0))
        return self.gpt2.generate(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
