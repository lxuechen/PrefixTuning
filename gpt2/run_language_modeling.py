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

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForData2TextLanguageModeling,  # modified
    HfArgumentParser,
    LineByLineData2TextTextDataset,  # modified
    LineByLineTriplesTextDataset,  # modified
    LineByLineWebNLGTextDataset,  # modified
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    set_seed,
    GPT2LMHeadModel,
)

from lxuechen_utils import utils
import privacy_utils
from .annoying_args import DataTrainingArguments, ModelArguments, PrivacyArguments, TrainingArguments
from .train_control import PrefixTuning
from .trainer_prefix import Trainer_Prefix
from . import prefix_tuning_minimal

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    cache_dir: Optional[str] = None,
    **_,
):
    file_path = {
        "train": args.train_data_file,
        "val": args.val_data_file,
        "eval": args.eval_data_file,
    }[split]
    if file_path is None:
        return None

    if args.line_by_line:
        if args.task_mode == 'data2text':
            dataset = LineByLineData2TextTextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token,
                max_seq_len=args.max_seq_len
            )
        elif args.task_mode == 'triples':
            dataset = LineByLineTriplesTextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size, bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token
            )
        elif args.task_mode == 'webnlg':
            dataset = LineByLineWebNLGTextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token
            )
        else:
            raise ValueError(f"Unknown `args.task_mode`: {args.task_mode}")

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
    train_dataset, val_dataset, eval_dataset = tuple(
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, split=split)
        for split in ("train", "val", "eval")
    )
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        if data_args.task_mode == 'data2text' or data_args.task_mode == 'triples' or data_args.task_mode == 'webnlg':
            data_collator = DataCollatorForData2TextLanguageModeling(
                tokenizer=tokenizer,
                mlm=data_args.mlm,
                mlm_probability=data_args.mlm_probability,
                format_mode=data_args.format_mode
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=data_args.mlm,
                mlm_probability=data_args.mlm_probability
            )
    return train_dataset, val_dataset, eval_dataset, data_collator


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

    # TODO: `config.objective_mode` matters a lot!
    config.return_dict = True
    config.objective_mode = model_args.objective_mode

    # `bos_token` and `eos_token` is the same for GPT-2; both are 50256.
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

    if model_args.tuning_mode == 'prefixtune':
        model = prefix_tuning_minimal.PrefixTuningMinimal(
            model_args=model_args, config=config,
        )
    elif model_args.tuning_mode == "fulltune":
        model = GPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError(f"Unknown tuning mode: {model_args.tuning_mode}")

    # 0 means the regular token level objective, which is sum / output_len
    # 1 means the sentence level objective, which is sum
    # 2 means our buggy version which is sum/max_batch(input_len +output_len)
    # 3 means our buggy version which is sum/max_batch(output_len)
    # 4 means our buggy version which is sum/(input_len +output_len)

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

    train_dataset, val_dataset, eval_dataset, data_collator = get_dataset_wrapper(
        config=config, tokenizer=tokenizer,
        data_args=data_args, training_args=training_args, model_args=model_args,
    )
    if model_args.tuning_mode == 'prefixtune':
        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
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

    # One-shot debugging stmts.
    # if False:
    #     train_loader = trainer.get_train_dataloader()
    #     batch = next(iter(train_loader))
    #     input_ids = batch["input_ids"]
    #     labels = batch["labels"]
    #     print(input_ids[0], labels[0])
    #     print(input_ids[1], labels[1])
    #     import pdb; pdb.set_trace()
    #     exit()

    if privacy_args.nonprivate == "no":
        # TODO: Why does the per_example_max_grad_norm not affect things by much???
        actual_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        privacy_engine = privacy_utils.privacy_engine.PrivacyEngine(
            module=model,

            # Privacy specific arguments.
            batch_size=actual_batch_size,  # This determines what's dividing the gradient!
            sample_size=len(train_dataset),
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            loss_reduction="mean",
            batch_first=True,
            accounting_mode=privacy_args.accounting_mode,
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
