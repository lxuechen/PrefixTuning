# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import glob
import logging
import math
import os
from typing import Optional

import torch
import transformers
from transformers.file_utils import cached_path

from annoying_args import DataTrainingArguments, ModelArguments
from train_control import ClassificationHead, PrefixTuning
from train_control2 import PrefixEmbTuning

path = os.path.abspath(transformers.__file__)

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWeightedLanguageModeling,  # modified
    DataCollatorForEmbMatchLanguageModeling,  # modified
    # modified
    # modified
    DataCollatorForKeywordLanguageModeling,  # modified
    DataCollatorForData2TextLanguageModeling,  # modified
    DataCollatorForText2DataLanguageModeling,  # modified
    DataCollatorForWritingPromptsLanguageModeling,  # modified
    DataCollatorForClassificationSentimentLanguageModeling,  # modified
    DataCollatorForSumLanguageModeling,  # modified
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithWeightTextDataset,  # modified
    LineByLineEmbMatchTextDataset,  # modified
    LineByLineTopicTextDataset,  # modified
    LineByLineKeywordTextDataset,  # modified
    LineByLineLengthTextDataset,  # modified
    LineByLineData2TextTextDataset,  # modified
    LineByLineLemma2TextTextDataset,  # modified
    LineByLineText2DataTextDataset,  # modified
    LineByLineTriplesTextDataset,  # modified
    LineByLineWebNLGTextDataset,  # modified
    LineByLineWritingPromptsTextDataset,  # modified
    LineByLineSentimentTextDataset,  # modified
    LineByLineClassificationSentimentTextDataset,  # modified
    LineByLineClassificationTopicTextDataset,
    LineByLineSumTextDataset,  # modified
    PreTrainedTokenizer,
    TextDataset,
    Trainer_Prefix,
    TrainingArguments,
    set_seed,
    GPT2LMHeadModel,
    GPT2LMHeadModelAdapter,
)

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
    "length": {
        "path": "/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/pplm"
                "/length_classifier_head_epoch_10.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very short": 0, "short": 1, "medium": 2, "long": 3, "very long": 4},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    }
}

logger = logging.getLogger(__name__)


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    training_args: TrainingArguments = None,
    finetune_mode: bool = False,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        if args.task_mode == 'embMatch':
            dataset = LineByLineEmbMatchTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                    block_size=args.block_size,
                                                    num_layer=1, bos_tok=tokenizer.bos_token,
                                                    eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'topic':
            dataset = LineByLineTopicTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                 block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                 eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'length':
            dataset = LineByLineLengthTextDataset(
                tokenizer=tokenizer, file_path=file_path,
                block_size=args.block_size, bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'keyword':
            dataset = LineByLineKeywordTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'data2text':
            dataset = LineByLineData2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                     block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token,
                                                     lowdata_token=args.lowdata_token if (
                                                         'lowdata' in training_args.output_dir and finetune_mode)
                                                     else None)

        elif args.task_mode == 'triples':
            dataset = LineByLineTriplesTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                   block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                   eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'webnlg':
            dataset = LineByLineWebNLGTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                  block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                  eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'writingPrompts':
            dataset = LineByLineWritingPromptsTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                          block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                          eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'cnndm' or args.task_mode == 'xsum':
            max_source_length = args.max_source_length
            max_target_length = args.train_max_target_length if not evaluate else args.val_max_target_length
            dataset = LineByLineSumTextDataset(tokenizer=tokenizer, file_path=file_path,
                                               block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                               eos_tok=tokenizer.eos_token, max_source_length=max_source_length,
                                               max_target_length=max_target_length, )

        elif args.task_mode == 'sentiment':
            dataset = LineByLineSentimentTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                     block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'classify-sentiment':
            dataset = LineByLineClassificationSentimentTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                                   block_size=args.block_size,
                                                                   bos_tok=tokenizer.bos_token,
                                                                   eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'classify-topic':
            dataset = LineByLineClassificationTopicTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                               block_size=args.block_size,
                                                               bos_tok=tokenizer.bos_token,
                                                               eos_tok=tokenizer.eos_token)

        elif args.task_mode == 'lemma2text':
            dataset = LineByLineLemma2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                      block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                      eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'text2data':
            dataset = LineByLineText2DataTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                     block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                     eos_tok=tokenizer.eos_token)
        elif args.task_mode == 'gen_data':
            dataset = LineByLineWithWeightTextDataset(tokenizer=tokenizer, file_path=file_path,
                                                      block_size=args.block_size, bos_tok=tokenizer.bos_token,
                                                      eos_tok=tokenizer.eos_token)
        else:
            return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        return dataset
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            cache_dir=cache_dir,
        )


def get_classifier(name: Optional[str], class_label: int, device: str):
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified in the discriminator model parameters")
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.disable_tqdm = False

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use "
            f"--overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from "
            "another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    config._my_arg_tune_mode = model_args.tuning_mode

    # 0 means the regular token level objective, which is sum / output_len
    # 1 means the sentence level objective, which is sum
    # 2 means our buggy version which is sum/max_batch(input_len +output_len)
    # 3 means our buggy version which is sum/max_batch(output_len)
    # 4 means our buggy version which is sum/(input_len +output_len)
    config._objective_mode = model_args.objective_mode
    config._my_arg_task_mode = data_args.task_mode

    if model_args.tuning_mode in ['finetune', 'adaptertune', 'finetune-top']:
        print('objective is 0 because of finetune')
    elif model_args.tuning_mode == 'prefixtune':
        print('objective is {}'.format(config._objective_mode))

    if model_args.tuning_mode == 'adaptertune':
        config.adapter_design = model_args.adapter_design
        config.bottleneck = model_args.adapter_bottleneck

        if model_args.model_name_or_path:
            config.return_dict = True
            model = GPT2LMHeadModelAdapter.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelWithLMHead.from_config(config)

    else:
        if model_args.model_name_or_path:
            print(config.return_dict)
            config.return_dict = True
            model = GPT2LMHeadModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelWithLMHead.from_config(config)

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    print(model_args.tuning_mode)
    print('adapting the size of the model embedding to include [PAD]')
    print('len(tokenizer) = ', len(tokenizer))
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    print('len(tokenizer) = ', len(tokenizer))
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.bos_token, tokenizer.bos_token_id)

    if model_args.tuning_mode == 'prefixtune':  # prefixtune
        for param in model.base_model.parameters():
            param.requires_grad = False

        gpt2 = model
        print('loading the prefix model from ', model_args.prefixModel_name_or_path)
        optim_prefix_bool: bool = model_args.optim_prefix.lower() == "yes"

        if model_args.prefixModel_name_or_path is not None:
            config2 = AutoConfig.from_pretrained(model_args.prefixModel_name_or_path, cache_dir=model_args.cache_dir)
            if model_args.prefix_mode == 'embedding':
                model = PrefixEmbTuning.from_pretrained(
                    model_args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                    config=config2,
                    cache_dir=model_args.cache_dir,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                    use_infix=(data_args.format_mode == 'infix')
                )
            elif model_args.prefix_mode == 'activation':
                model = PrefixTuning.from_pretrained(
                    model_args.prefixModel_name_or_path,
                    from_tf=bool(".ckpt" in model_args.prefixModel_name_or_path),
                    config=config2,
                    cache_dir=model_args.cache_dir,
                    model_gpt2=gpt2, optim_prefix=optim_prefix_bool, preseqlen=model_args.preseqlen,
                    use_infix=(data_args.format_mode == 'infix')
                )
            else:
                assert False, "invalid prefix mode"
        else:
            # should clone the config and construct it.
            config_prefix = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
            config_prefix._my_arg_tune_mode = model_args.tuning_mode
            config_prefix._my_arg_task_mode = data_args.task_mode
            config_prefix._my_arg_control = True
            config_prefix.train_weights = data_args.train_embs
            config_prefix.optim_prefix = optim_prefix_bool
            config_prefix.preseqlen = model_args.preseqlen
            config_prefix.use_infix = (data_args.format_mode == 'infix')
            config_prefix.format_mode = data_args.format_mode
            config_prefix.prefix_dropout = model_args.prefix_dropout
            config_prefix.vocab_size = len(tokenizer)
            config_prefix.lowdata = ('lowdata' in training_args.output_dir)
            if config_prefix.lowdata and data_args.use_lowdata_token == 'yes':
                config_prefix.lowdata_token = tokenizer([data_args.lowdata_token],
                                                        add_prefix_space=True)['input_ids']  # return_tensors='np',
                print(data_args.lowdata_token)
                print(config_prefix.lowdata_token)

            # some extra stuff.
            config_prefix.init_random = model_args.init_random
            config_prefix.mid_dim = model_args.mid_dim

            print('training the prefix model from scratch. ')
            if model_args.prefix_mode == 'embedding':
                config_prefix.parametrize_emb = model_args.parametrize_emb
                model = PrefixEmbTuning(config_prefix, model_gpt2=gpt2)
            elif model_args.prefix_mode == 'activation':
                # TODO: Model is created here!
                print('model created here!')
                model = PrefixTuning(config_prefix, model_gpt2=gpt2)
            else:
                assert False, "invalid prefix mode"

        print('Not in dataless setting, loading the control code. ')
        if 'sentiment' in training_args.output_dir:
            print('sentiment does need discri_labels')
            discri_labels = None
        elif 'classify-sentiment' in training_args.output_dir:
            print('classify-sentiment does need discri_labels')
            discri_labels = None
        elif 'classify-topic' in training_args.output_dir:
            print('classify-topic does need discri_labels')
            discri_labels = None
        elif 'sent' in training_args.output_dir:
            discri_labels = ['negative', 'positive']
        elif 'topic' in training_args.output_dir:
            discri_labels = ['world', 'sports', 'business', 'science']
        elif 'keyword' in training_args.output_dir:
            print('keyword is unbounded.')
            discri_labels = None
        elif 'embMatch' in training_args.output_dir:
            print('embMatch is unbounded.')
            discri_labels = None
        elif 'data2text' in training_args.output_dir:
            print('data2text does need discri_labels')
            discri_labels = None
        elif 'triples' in training_args.output_dir:
            print('triples does need discri_labels')
            discri_labels = None
        elif 'webnlg' in training_args.output_dir:
            print('triples does need discri_labels')
            discri_labels = None
        elif 'writingPrompts' in training_args.output_dir:
            print('writingPrompts does need discri_labels')
            discri_labels = None
        elif 'cnndm' in training_args.output_dir:
            print('cnndm does need discri_labels')
            discri_labels = None
        elif 'xsum' in training_args.output_dir:
            print('xsum does need discri_labels')
            discri_labels = None
        elif 'lemma2text' in training_args.output_dir:
            print('lemma2text does need discri_labels')
            discri_labels = None
        else:
            assert False, 'should have topic/sent in the file name'

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, training_args=training_args,
                    finetune_mode=(model_args.tuning_mode == 'finetune')) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir,
                    training_args=training_args, finetune_mode=(model_args.tuning_mode == 'finetune'))
        if training_args.do_eval
        else None
    )
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        if data_args.task_mode == 'embMatch':
            data_collator = DataCollatorForEmbMatchLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        elif data_args.task_mode == 'topic' or data_args.task_mode == 'sentiment':
            data_collator = DataCollatorForKeywordLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        elif data_args.task_mode == 'classify-topic' or data_args.task_mode == 'classify-sentiment':
            data_collator = DataCollatorForClassificationSentimentLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        elif data_args.task_mode == 'length':
            data_collator = DataCollatorForKeywordLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        elif data_args.task_mode == 'keyword':
            data_collator = DataCollatorForKeywordLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        elif data_args.task_mode == 'data2text' or data_args.task_mode == 'triples' or data_args.task_mode == \
            'webnlg':
            print('FORMAT MODE IS ', data_args.format_mode)
            data_collator = DataCollatorForData2TextLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
                format_mode=data_args.format_mode
            )
        elif data_args.task_mode == 'writingPrompts':
            print('FORMAT MODE IS ', data_args.format_mode)
            data_collator = DataCollatorForWritingPromptsLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
                format_mode=data_args.format_mode
            )
        elif data_args.task_mode == 'xsum' or data_args.task_mode == 'cnndm':
            print('FORMAT MODE IS ', data_args.format_mode)
            data_collator = DataCollatorForSumLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
                format_mode=data_args.format_mode
            )
        elif data_args.task_mode == 'lemma2text':
            data_collator = DataCollatorForData2TextLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        elif data_args.task_mode == 'text2data':
            data_collator = DataCollatorForText2DataLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        elif data_args.task_mode == 'gen_data':
            data_collator = DataCollatorForWeightedLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )

    if (model_args.tuning_mode == 'prefixtune'):
        if 'topic' in training_args.output_dir:
            discri_labels = ['world', 'sports', 'business', 'science']
        elif 'sent' in training_args.output_dir:
            discri_labels = ['negative', 'positive']
        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
            discri_labels=discri_labels,
            model_gpt2=gpt2,
            args=training_args,
            prediction_loss_only=True,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            task_mode=data_args.task_mode,
            use_dropout=(model_args.use_dropout == 'yes')
        )
    else:
        raise ValueError(f"Unsupported tuning_mode: {model_args.tuning_mode}")

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        trainer.train(model_path=model_path)
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval and not (data_args.dataless == 'yes'):
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    if data_args.task_mode == 'data2text':
        del model
        del trainer
        if model_args.tuning_mode == 'prefixtune' or model_args.tuning_mode == 'bothtune':
            del gpt2
        torch.cuda.empty_cache()
        elem = os.path.abspath(training_args.output_dir)
        checkpoint_path = elem

        print('running evaluation on ', checkpoint_path)

        os.system('python ../text-generation/gen.py data2text yes yes {} no'.format(checkpoint_path))

        if 'earlystop' in training_args.output_dir:
            elem = os.path.abspath(training_args.output_dir)
            checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
            assert len(checkpoint_path) == 1
            checkpoint_path = checkpoint_path[0]

            print('running early stopping evaluation on ', checkpoint_path)

            os.system('python ../text-generation/gen.py data2text yes yes {} no'.format(checkpoint_path))


    elif data_args.task_mode == 'webnlg':
        del model
        del trainer
        if model_args.tuning_mode == 'prefixtune':
            del gpt2
        torch.cuda.empty_cache()
        elem = os.path.abspath(training_args.output_dir)
        checkpoint_path = elem

        print('running evaluation on ', checkpoint_path)

        os.system('python ../text-generation/gen.py webnlg yes yes {} no'.format(checkpoint_path))

        # also run for early stopping:
        if 'earlystop' in training_args.output_dir:
            elem = os.path.abspath(training_args.output_dir)
            checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
            assert len(checkpoint_path) == 1
            checkpoint_path = checkpoint_path[0]

            print('running early stopping evaluation on ', checkpoint_path)

            os.system('python ../text-generation/gen.py webnlg yes yes {} no'.format(checkpoint_path))


    elif data_args.task_mode == 'triples':
        del model
        del trainer
        if model_args.tuning_mode == 'prefixtune':
            del gpt2
        torch.cuda.empty_cache()
        elem = os.path.abspath(training_args.output_dir)
        checkpoint_path = elem

        print('running evaluation on ', checkpoint_path)

        os.system('python ../text-generation/gen.py triples yes yes {} no'.format(checkpoint_path))

        if 'earlystop' in training_args.output_dir:
            elem = os.path.abspath(training_args.output_dir)
            checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
            assert len(checkpoint_path) == 1
            checkpoint_path = checkpoint_path[0]

            print('running early stopping evaluation on ', checkpoint_path)

            os.system('python ../text-generation/gen.py triples yes yes {} no'.format(checkpoint_path))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
