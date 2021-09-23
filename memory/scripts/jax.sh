#!/bin/bash

cache_dir="/nlp/scr/lxuechen/prefixtune/memory/cache"
micro_batch_size=25
gradient_accumulation_steps=2
num_updates=2
seq_len=100
model_name_or_path="gpt2"

python -m memory.jax_dp_grad_accumulation \
  --batch_size ${micro_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --seq_len ${seq_len} \
  --model_name_or_path ${model_name_or_path} \
  --num_updates ${num_updates} \
  --cache_dir ${cache_dir}
