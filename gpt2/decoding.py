"""Decoding for prefix-tuning."""

import torch
from transformers import HfArgumentParser, AutoConfig, AutoTokenizer, GPT2LMHeadModel

from . import prefix_tuning_minimal
from . import run_language_modeling
from .annoying_args import DataTrainingArguments, ModelArguments, PrivacyArguments, TrainingArguments
from .trainer_prefix import Trainer_Prefix


def create_essentials(model_args):
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config.return_dict = True
    config.objective_mode = model_args.objective_mode

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    gpt2 = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir,
    )
    gpt2.resize_token_embeddings(len(tokenizer))

    model = prefix_tuning_minimal.PrefixTuningMinimal(
        model_args=model_args, config=config, gpt2=gpt2,
    )
    return dict(config=config, tokenizer=tokenizer, gpt2=gpt2, model=model)


def main(
):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PrivacyArguments))
    model_args, data_args, training_args, privacy_args = parser.parse_args_into_dataclasses()

    essentials = create_essentials(model_args)
    config, tokenizer, gpt2, model = (
        essentials["config"], essentials["tokenizer"], essentials["gpt2"], essentials["model"]
    )

    # TODO: This block_size thing is very hacky and prone to error.
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # @formatter:off
    # Good checkpoint directory.
    ckpt_path = (
        "/nlp/scr/lxuechen/prefixtune/date_0620" 
        "/model_name_distilgpt2_nonprivate_no_tuning_mode_prefixtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_70000000_learning_rate_0_00100000_train_batch_size_00000100_mid_dim_00000512_preseqlen_00000010/0/checkpoint-15000/pytorch_model.bin"
    )
    # @formatter:on

    # Load the prefix checkpoint.
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    print(model)

    train_dataset, val_dataset, eval_dataset, data_collator = run_language_modeling.get_dataset_wrapper(
        config=config, tokenizer=tokenizer,
        data_args=data_args, training_args=training_args, model_args=model_args,
    )

    generation_stuff = dict(
        train_prompts=run_language_modeling.get_prompt_dataset(
            file_path=data_args.train_prompt_file, tokenizer=tokenizer
        ),
        val_prompts=run_language_modeling.get_prompt_dataset(
            file_path=data_args.val_prompt_file, tokenizer=tokenizer
        ),
        eval_prompts=run_language_modeling.get_prompt_dataset(
            file_path=data_args.eval_prompt_file, tokenizer=tokenizer
        ),
    )

    if model_args.tuning_mode == 'prefixtune':
        trainer = Trainer_Prefix(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            task_mode=data_args.task_mode,
            use_dropout=(model_args.use_dropout == 'yes'),
            generation_stuff=generation_stuff,
        )
    else:
        raise ValueError

    trainer.generate_and_write_to_file()


if __name__ == "__main__":
    # python -m gpt2.decoding
    main()
