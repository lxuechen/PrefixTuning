import torch
from torch import nn
from transformers import GPT2PreTrainedModel


class PrefixTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""

    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False):
        super().__init__(config)
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # TODO: What's this arg doing???
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
            self.task_mode = 'underspecified'
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

        if hasattr(config, 'init_shallow'):
            self.init_shallow = (config.init_shallow == 'yes')
        else:
            self.init_shallow = False

        if hasattr(config, 'init_shallow_word'):
            self.init_shallow_word = config.init_shallow_word
        else:
            self.init_shallow_word = None

        if True:
            self.mode_para = 0
            print('PrefixTuning')
            print('preseqlen is {}, optimizing the prefix directly'.format(self.preseqlen))
            if self.lowdata and self.lowdata_token is not None:
                low_data_init = 3
                # use a single prepended token.
                assert self.lowdata_token is not None
                self.preseqlen = len(self.lowdata_token[0])
                print('LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, '
                      'preseqlen = {} Unifying with FINETUNE'.format(self.preseqlen))
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                self.get_prompt = self.get_prompt_p5


            # DIFFERENT PARAMETRIZATION:
            elif not deep_param and not self.init_shallow:
                # TODO: Marker --- this by default!
                low_data_init = 0
                print('[Full prefix-tuning Setting :) ]')
                self.register_buffer('input_tokens', torch.arange(self.preseqlen))
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd)
                )
                self.get_prompt = self.get_prompt_p5

            elif self.init_shallow:
                low_data_init = 0
                print('[DOUBLE CHECK]: ABLATION STUDY on no parametrization trick... [shallow]')

                if self.init_shallow_word is not None:
                    assert self.init_shallow_word is not None
                    self.preseqlen = len(self.init_shallow_word[0])
                    # init it by the init_shallow_word
                    init_val = self.get_gold_init(model_gpt2, torch.LongTensor(self.init_shallow_word))
                    print(init_val.shape)
                    self.control_trans = nn.Parameter(init_val)

                    # torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p2_shallow
                else:
                    print('random init of the prefix')
                    self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p2

            else:
                low_data_init = 0
                print('[DOUBLE CHECK]: DEEP MLP')
                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                self.get_prompt = self.get_prompt_p5

        self.dropout = nn.Dropout(self.prefix_dropout)

        total_param = sum(param.numel() for param in self.parameters())
        print('total param is {} million'.format(total_param / 1e6))

        if low_data_init == 3:
            print('use pt for this tensor', torch.LongTensor(self.lowdata_token))
            self.lowdata_init_train3(gpt2=model_gpt2, sample_input=torch.LongTensor(self.lowdata_token))

    def get_gold_init(self, gpt2, sample_input):
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            output = gpt2(sample_input.to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)
        return output

    def lowdata_init_train3(self, gpt2, sample_input, epochs=500):  # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            output = gpt2(sample_input.to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()
        return

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        temp_control = self.control_trans.view(
            1, self.preseqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        ).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p2_shallow(self, control_code=None, gpt2=None, bsz=None):
        temp = self.control_trans.expand(-1, bsz, -1, -1, -1)
        return temp.split(2)

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        # TODO: Marker --- this by default
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
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
                gpt2_model,
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
                src=None,
                tgt=None,
                src_attn=None,
                tgt_attn=None,
                **kwargs):
        del past_key_values
        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        else:
            past_key_values_prompt = self.get_prompt(control_code, gpt2=gpt2_model, bsz=bsz)
        past_key_values = past_key_values_prompt

        # TODO: When is this concatenation used?
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
            return_dict=return_dict,
            **kwargs
        )
        return output
