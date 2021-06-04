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
        print(f"self.optim_prefix: {self.optim_prefix}")

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen
        print(f"self.preseqlen: {self.preseqlen}")

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix
        print(f"self.use_infix: {self.use_infix}")

        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'
        print(f"self.tuning_mode: {self.tuning_mode}")

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            assert False, 'the task is underspecified'
        print(f"self.task_mode: {self.task_mode}")

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'
        print(f"self.format_mode: {self.format_mode}")

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0
        print(f"self.prefix_dropout: {self.prefix_dropout}")

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False
        print(f"self.init_random: {self.init_random}")

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512
        print(f"self.mid_dim: {self.mid_dim}")

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

    def lowdata_init_train1(self, gpt2, tokenizer, sample_input):
        input = tokenizer(sample_input, return_tensors='pt')
        output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
        output = output.past_key_values
        print(len(output), output[0].shape)
        output = torch.cat(output, dim=0).detach()
        return torch.nn.Parameter(output)

    def get_prompt_p22(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        past_key_values = self.control_trans.expand(-1, bsz, -1, -1, -1).split(2, dim=0)
        return past_key_values

    def lowdata_init_train2(self, gpt2, tokenizer, sample_input, epochs=500):  # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            input = tokenizer(sample_input, return_tensors='pt')
            output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
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
        temp_control = self.control_trans.view(1, self.preseqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p3_infix(self, src, control_code=None, gpt2=None, bsz=None):
        # temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        # print('infix')
        src_out = gpt2(input_ids=src, use_cache=True, return_dict=True, output_hidden_states=True)
        src_repr = src_out.hidden_states[-1]  # bsz, seqlen, hidden
        src_past_key_vals = src_out.past_key_values
        past_key_values = self.control_trans(src_repr)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        # print(past_key_values.shape)
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        full_lst = []
        for i in range(len(src_past_key_vals)):
            full_lst.append(torch.cat([src_past_key_vals[i], past_key_values[i]], dim=3))

        return full_lst

    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
        return past_key_values

    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p5_infix(self, src, control_code=None, gpt2=None, bsz=None, attn_mask=None):
        # VERSION1. infixing by taking in the last layer of the hidden states as input.

        # VERSION2. infixing by pretending some input to first get the history, then add upon them.
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])

        temp_emb = self.wte2(input_tokens)
        src_emb = gpt2.transformer.wte(src)
        total_emb = torch.cat([src_emb, temp_emb], dim=1)  # bsz, seqlen, dim
        src_out = gpt2(inputs_embeds=total_emb, attention_mask=attn_mask, use_cache=True, return_dict=True)
        src_past_key_vals = src_out.past_key_values
        src_past_key_vals = torch.cat(src_past_key_vals, dim=0)
        # print(src_past_key_vals.shape, past_key_values.shape) # the src should be longer than past.
        # get a zero mask.
        _, src_len = src.shape
        nl, nb, nh, _, ndim = past_key_values.shape
        zero_mask = torch.zeros(nl, nb, nh, src_len, ndim).to(self.device)
        # print(zero_mask.shape, past_key_values.shape)
        past_key_values = torch.cat([zero_mask, past_key_values], dim=3)
        # print(past_key_values.shape)
        past_key_values = past_key_values + src_past_key_vals

        # add them together.
        past_key_values = past_key_values.split(2)

        return past_key_values

    def get_prompt_p6(self, control_code=None, gpt2=None, bsz=None):
        input_embs = self.input_embs.to(self.device)
        past_key_values = self.control_trans(input_embs).expand(bsz, -1, -1)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code)  # bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control).mean(1).unsqueeze(1)  # bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def get_prompt_p1(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:

            if type(control_code) is tuple:
                assert False, 'Tuples'
                control_embs, control_word = control_code
                past_key_values = self.control_trans(control_embs)
                past_key_values = past_key_values.mean(1).unsqueeze(1)
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen * self.preseqlen, self.match_n_layer * 2,
                                                       self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                print(control_word, control_embs.shape)
            else:
                # print('running with control code')
                # use the control code to generate the first 5 activation layers.
                if not self.embMatch:
                    if self.wte:
                        temp_control = self.wte(control_code)
                    else:
                        assert gpt2 is not None
                        temp_control = gpt2.transformer.wte(control_code)
                    temp_control = temp_control.sum(1).unsqueeze(1)
                else:
                    temp_control = control_code
                    # print(control_code.shape)
                past_key_values = self.control_trans(temp_control)
                # print(past_key_values.shape) #bsz, controlCodeLen, long... 5 * config.n_layer * 2 * config.n_embd
                past_key_values = past_key_values.sum(1).unsqueeze(1)
                # print(past_key_values.shape)  # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen * self.preseqlen, self.match_n_layer * 2,
                                                       self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
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
                **kwargs,
                ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

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
        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                            head_mask=head_mask, inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict, **kwargs)

        return output

    def forward_infix(self,
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
                      cate_batch=None,
                      cate_attn=None,
                      **kwargs,
                      ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1)  # bsz, seqlen
        else:
            infix_attn = torch.ones(bsz, self.preseqlen).bool().to(self.device)
            attention_mask = torch.cat([src_attn, infix_attn, tgt_attn], dim=1)  # bsz, seqlen
            partial_attn_mask = torch.cat([src_attn, infix_attn], dim=1)  # bsz, seqlen
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz, attn_mask=partial_attn_mask)
            # print(src_attn)
            # print()
            # print(infix_attn)
            # infix_attn = torch.ones(bsz, self.preseqlen).to(self.device)
            # attention_mask = torch.cat([src_attn, infix_attn, tgt_attn], dim=1)  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                            head_mask=head_mask, inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict, **kwargs)

        return output
