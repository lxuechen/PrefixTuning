#!/bin/bash
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-2-64 
CUDA_VISIBLE_DEVICES=0 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode fulltune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-2-64" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-2-64 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-2-64/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-4-128 
CUDA_VISIBLE_DEVICES=1 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode fulltune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-4-128" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-4-128 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-4-128/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-6-192 
CUDA_VISIBLE_DEVICES=2 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode fulltune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-6-192" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-6-192 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-6-192/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-8-256 
CUDA_VISIBLE_DEVICES=3 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode fulltune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-8-256" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-8-256 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-8-256/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-10-320 
CUDA_VISIBLE_DEVICES=4 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode fulltune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-10-320" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-10-320 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-10-320/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-12-384 
CUDA_VISIBLE_DEVICES=5 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode fulltune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-12-384" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-12-384 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-12-384/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-14-448 
CUDA_VISIBLE_DEVICES=6 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode fulltune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-14-448" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-14-448 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/fulltune/distilgpt2-14-448/log.out & 
wait
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-2-64 
CUDA_VISIBLE_DEVICES=0 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode scratchtune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-2-64" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-2-64 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-2-64/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-4-128 
CUDA_VISIBLE_DEVICES=1 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode scratchtune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-4-128" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-4-128 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-4-128/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-6-192 
CUDA_VISIBLE_DEVICES=2 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode scratchtune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-6-192" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-6-192 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-6-192/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-8-256 
CUDA_VISIBLE_DEVICES=3 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode scratchtune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-8-256" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-8-256 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-8-256/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-10-320 
CUDA_VISIBLE_DEVICES=4 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode scratchtune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-10-320" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-10-320 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-10-320/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-12-384 
CUDA_VISIBLE_DEVICES=5 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode scratchtune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-12-384" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-12-384 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-12-384/log.out & 
mkdir -p /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-14-448 
CUDA_VISIBLE_DEVICES=6 python -m gpt2stuff.launchers.prefix_vs_full_062021 --mode "local" --tuning_mode scratchtune --max_seq_len 100 --nonprivate "no" --per_device_train_batch_size 32 --gradient_accumulation_steps 16 --learning_rate 0.0005 --per_example_max_grad_norm 0.1 --target_epsilon 8 --epochs 50 --private_engine_mode "ghost"   --model_name_or_path "/home/lxuechen_stanford_edu/dump/date_080221/distilgpt2-14-448" --train_dir /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-14-448 |& tee /nlp/scr/lxuechen/prefixtune/date_080321/scratchtune/distilgpt2-14-448/log.out & 
wait
