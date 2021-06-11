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

import logging
import os
from typing import Optional

import privacy_utils
from .train_control import PrefixTuning
from .trainer_prefix import Trainer_Prefix
from .annoying_args import DataTrainingArguments, ModelArguments, PrivacyArguments, TrainingArguments
from lxuechen_utils import utils


from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
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
    Trainer,
    set_seed,
    GPT2LMHeadModel,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
            dataset = LineByLineLengthTextDataset(tokenizer=tokenizer, file_path=file_path,
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
                                                     else None,
                                                     max_seq_len=args.max_seq_len)

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


def get_dataset_wrapper(config, tokenizer, data_args, model_args, training_args):
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, training_args=training_args,
                    finetune_mode=(model_args.tuning_mode == 'finetune'))
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
        elif data_args.task_mode == 'data2text' or data_args.task_mode == 'triples' or data_args.task_mode == 'webnlg':
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
    return train_dataset, eval_dataset, data_collator


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PrivacyArguments))
    model_args, data_args, training_args, privacy_args = parser.parse_args_into_dataclasses()

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
    # TODO: Double check this objective mode.
    config._objective_mode = model_args.objective_mode
    config._my_arg_task_mode = data_args.task_mode

    if model_args.tuning_mode in ['finetune', 'adaptertune', 'finetune-top']:
        print('objective is 0 because of finetune')
    elif model_args.tuning_mode == 'prefixtune':
        print('objective is {}'.format(config._objective_mode))

    config.return_dict = True
    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
    )

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    ##############################################################
    ################# ADJUST TOKENIZER ###########################
    ##############################################################

    print(model_args.tuning_mode)
    print('adapting the size of the model embedding to include [PAD]')
    print('len(tokenizer) = ', len(tokenizer))
    num_added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    print('len(tokenizer) = ', len(tokenizer))
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.bos_token, tokenizer.bos_token_id)

    if model_args.tuning_mode == 'prefixtune':
        gpt2 = model.requires_grad_(False)
        optim_prefix_bool = {"yes": True, "no": False}[model_args.optim_prefix]

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
        if not config_prefix.lowdata:
            config_prefix.lowdata = (
                'datalevels' in training_args.output_dir and data_args.use_lowdata_token == 'yes'
            )
        if config_prefix.lowdata and data_args.use_lowdata_token == 'yes':
            config_prefix.lowdata_token = tokenizer(
                [data_args.lowdata_token], add_prefix_space=True
            )['input_ids']
            print(data_args.lowdata_token)
            print(config_prefix.lowdata_token)

        # some extra stuff.
        config_prefix.init_random = model_args.init_random
        config_prefix.mid_dim = model_args.mid_dim
        config_prefix.init_shallow = model_args.init_shallow
        if config_prefix.init_shallow == 'yes':
            if model_args.init_shallow_word != 'no':
                config_prefix.init_shallow_word = tokenizer(
                    [model_args.init_shallow_word],
                    add_prefix_space=True
                )['input_ids']
            else:
                config_prefix.init_shallow_word = None
            print(model_args.init_shallow_word)
            print(config_prefix.init_shallow_word)

        model = PrefixTuning(config_prefix, model_gpt2=gpt2)
    elif model_args.tuning_mode == "fulltune":
        model.requires_grad_(True)
    else:
        raise ValueError(f"Unknown tuning mode: {model_args.tuning_mode}")

    train_dataset, eval_dataset, data_collator = get_dataset_wrapper(
        config=config, tokenizer=tokenizer,
        data_args=data_args, training_args=training_args, model_args=model_args,
    )
    if model_args.tuning_mode == 'prefixtune':
        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
            model_gpt2=gpt2,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            task_mode=data_args.task_mode,
            use_dropout=(model_args.use_dropout == 'yes'),
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)
    trainer.create_optimizer_and_scheduler(t_total)

    if privacy_args.nonprivate == "no":
        actual_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        privacy_engine = privacy_utils.privacy_engine.PrivacyEngine(
            batch_size=actual_batch_size,
            module=model,
            sample_size=len(train_dataset),
            epochs=training_args.num_train_epochs,
            # Privacy specific arguments.
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
        )
        privacy_engine.attach(trainer.optimizer)
        print(privacy_engine)

    # Training
    if training_args.do_train:
        all_args = {
            **training_args.__dict__,
            **data_args.__dict__,
            **model_args.__dict__,
            **privacy_args.__dict__,
        }
        utils.jdump(
            all_args,
            os.path.join(training_args.output_dir, 'argparse.json'),
            default=lambda x: str(x),
        )

        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        logger.info("*** Train ***")
        logger.info(
            f"Training set size: {len(train_dataset)}, "
            f"per_device_train_batch_size: {training_args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}"
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        if model_args.tuning_mode == 'bothtune':
            gpt2_dir = os.path.join(training_args.output_dir, 'gpt2')
            gpt2.save_pretrained(gpt2_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate(eval_dataset)
        eval_file = os.path.join(training_args.output_dir, "eval_results.json")
        utils.jdump(eval_output, eval_file)

        train_output = trainer.evaluate(train_dataset)
        train_file = os.path.join(training_args.output_dir, "train_results.json")
        utils.jdump(train_output, train_file)

        results = {
            "train_eval_loss": train_output["eval_loss"],
            "eval_lin_logprob": eval_output["lin_logprob"],
        }
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
