#!/bin/bash
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.001         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 2         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.001         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 5         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00100000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.001         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 8         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.0005         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 2         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.0005         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 5         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.0005         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 8         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000002/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.0001         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 2         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000005/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.0001         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 5         --overwrite_output_dir         --lr_decay yes'
mkdir -p "/nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0"
nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/log.out -p low --memory 16g --hold 'python -m gpt2stuff.run_language_modeling         --output_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0         --task_mode data2text         --model_type gpt2         --model_name_or_path distilgpt2         --tokenizer_name distilgpt2         --per_device_train_batch_size 16         --per_device_eval_batch_size 10         --save_steps 50000         --num_train_epochs 50         --do_train         --do_eval         --line_by_line         --save_total_limit 1         --data_folder /nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data         --tuning_mode lasttune         --logging_dir /nlp/scr/lxuechen/prefixtune/date_0718/model_name_distilgpt2_nonprivate_no_tuning_mode_lasttune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00010000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0         --logging_steps -1         --optim_prefix yes         --preseqlen 10         --prefix_mode activation         --format_mode cat         --gradient_accumulation_steps 32         --learning_rate 0.0001         --weight_decay 0.0         --seed 0         --mid_dim 512         --init_random no         --use_dropout no         --prefix_dropout 0.0         --objective_mode 0         --evaluate_during_training         --eval_steps 100         --eval_epochs 5         --noise_multiplier -1         --nonprivate no         --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/         --max_steps -1         --max_eval_batches 100         --evaluation_strategy epoch         --per_example_max_grad_norm 0.1         --max_seq_len 100         --max_generations 9223372036854775807         --max_generations_train 60         --ema_model_averaging no         --ema_model_start_from 1000         --efficient no         --debug no         --target_delta 1e-05         --target_epsilon 8         --overwrite_output_dir         --lr_decay yes'
