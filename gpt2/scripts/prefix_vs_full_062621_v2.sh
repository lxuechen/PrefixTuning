#!/bin/bash
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0626/model_name_distilgpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_80000000_learning_rate_0_00050000_train_batch_size_00000400_mid_dim_00000512_preseqlen_00000010_epochs_00000060_target_epsilon_00000008/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,jagupard14 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0626/model_name_distilgpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_80000000_learning_rate_0_00050000_train_batch_size_00000400_mid_dim_00000512_preseqlen_00000010_epochs_00000060_target_epsilon_00000008/0/log.out -p standard --memory 16g -d titanx 'python -m gpt2.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0626/model_name_distilgpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_80000000_learning_rate_0_00050000_train_batch_size_00000400_mid_dim_00000512_preseqlen_00000010_epochs_00000060_target_epsilon_00000008/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 20         --per_device_eval_batch_size 10         --save_steps 1000         --num_train_epochs 60         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --train_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt         --val_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt         --eval_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt         --tuning_mode scratchtune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0626/model_name_distilgpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_0_80000000_learning_rate_0_00050000_train_batch_size_00000400_mid_dim_00000512_preseqlen_00000010_epochs_00000060_target_epsilon_00000008/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 20         --learning_rate 0.0005         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 1000         --noise_multiplier 0.8         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy "steps"         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --train_prompt_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_train.txt         --val_prompt_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_valid.txt         --eval_prompt_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/prompts_test.txt         --ema_model_averaging yes         --ema_model_start_from 1000         --efficient yes         --target_delta 1e-05         --target_epsilon 8         --overwrite_output_dir'
