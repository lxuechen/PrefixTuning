#!/bin/bash
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,jagupard14 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0/log.out -p standard --memory 16g 'python -m gpt2.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0         --task_mode "data2text"         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 5         --per_device_eval_batch_size 25         --save_steps 500000         --num_train_epochs 5         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --train_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt         --val_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt         --eval_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt         --tuning_mode prefixtune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0         --logging_steps -1         --optim_prefix yes         --preseqlen 5         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 1         --learning_rate 1e-05         --weight_decay 0.0         --seed 0         --mid_dim 256         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 1         --evaluate_during_training         --eval_steps 100         --noise_multiplier 0.8         --nonprivate yes         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy "steps"         --per_example_max_grad_norm 1.0         --max_seq_len 96         --max_generations 9223372036854775807         --overwrite_output_dir'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,jagupard14 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0/log.out -p standard --memory 16g 'python -m gpt2.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0         --task_mode "data2text"         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 5         --per_device_eval_batch_size 25         --save_steps 500000         --num_train_epochs 5         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --train_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt         --val_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt         --eval_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt         --tuning_mode prefixtune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0         --logging_steps -1         --optim_prefix yes         --preseqlen 5         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 1         --learning_rate 1e-06         --weight_decay 0.0         --seed 0         --mid_dim 256         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 1         --evaluate_during_training         --eval_steps 100         --noise_multiplier 0.8         --nonprivate yes         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy "steps"         --per_example_max_grad_norm 1.0         --max_seq_len 96         --max_generations 9223372036854775807         --overwrite_output_dir'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,jagupard14 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0/log.out -p standard --memory 16g 'python -m gpt2.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0         --task_mode "data2text"         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 5         --per_device_eval_batch_size 25         --save_steps 500000         --num_train_epochs 5         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --train_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt         --val_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt         --eval_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt         --tuning_mode fulltune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00001000_train_batch_size_00000005_mid_dim_00000256/0         --logging_steps -1         --optim_prefix yes         --preseqlen 5         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 1         --learning_rate 1e-05         --weight_decay 0.0         --seed 0         --mid_dim 256         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 1         --evaluate_during_training         --eval_steps 100         --noise_multiplier 0.8         --nonprivate yes         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy "steps"         --per_example_max_grad_norm 1.0         --max_seq_len 96         --max_generations 9223372036854775807         --overwrite_output_dir'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,jagupard14 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0/log.out -p standard --memory 16g 'python -m gpt2.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0         --task_mode "data2text"         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 5         --per_device_eval_batch_size 25         --save_steps 500000         --num_train_epochs 5         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --train_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_train.txt         --val_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_valid.txt         --eval_data_file /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/src1_test.txt         --tuning_mode fulltune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0617/model_name_distilgpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00000100_train_batch_size_00000005_mid_dim_00000256/0         --logging_steps -1         --optim_prefix yes         --preseqlen 5         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 1         --learning_rate 1e-06         --weight_decay 0.0         --seed 0         --mid_dim 256         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 1         --evaluate_during_training         --eval_steps 100         --noise_multiplier 0.8         --nonprivate yes         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy "steps"         --per_example_max_grad_norm 1.0         --max_seq_len 96         --max_generations 9223372036854775807         --overwrite_output_dir'
