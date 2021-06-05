import torch
from torch import nn
from transformers import GPT2PreTrainedModel


class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        return self.mlp(hidden_state)


class PrefixTuning(GPT2PreTrainedModel):

    def __init__(
        self,
        config,
        optim_prefix=False,
        preseqlen=5,
        use_infix=False,
        **kwargs,
    ):
        super().__init__(config)
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None

        self.mode_para = 0
        print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
        print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))
        print('UNDER PARAMETRIZATION 1')
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, config.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(config.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd),
        )
        self.get_prompt = self.get_prompt_p5
        self.dropout = nn.Dropout(self.prefix_dropout)

        trainable_params = sum(param.numel() for param in self.parameters())
        print(f"total trainable param {trainable_params / 1e6:.4f} million")

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
                input_ids=None,
                weights=None,
                control_code=None,
                emb_match=None,
                past_key_values=None,
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
                gpt2_model=None,
                src=None,
                tgt=None,
                src_attn=None,
                tgt_attn=None,
                **kwargs):
        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)
        output = gpt2_model(
            input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
            past_key_values=past_key_values, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, **kwargs
        )
        return output
