## For playing around locally on E2Es

```bash
python -m gpt2.launchers.prefix_vs_full_062021 --mode "local"     \
  --tuning_mode "prefixtune"      \
  --max_steps 10000 \
  --max_seq_len 96 \
  --nonprivate "no"\
  --per_device_train_batch_size 5 \
  --gradient_accumulation_steps 1 \
  --noise_multiplier 0.7 \
  --eval_steps 100 \
  --objective_mode 0 \
  --max_generations 40 \
  --learning_rate 1e-5 \
  --per_example_max_grad_norm 0.1 \
  --mid_dim 512 \
  --preseqlen 10
```

After refactor on July 2, 2021 (rename folder `gpt2` with `gpt2stuff`)

```bash
python -m gpt2stuff.launchers.prefix_vs_full_062021 \
  --mode "local" \
  --tuning_mode "fulltune" \
  --max_seq_len 100 \
  --nonprivate "no"\
  --per_device_train_batch_size 20 \
  --gradient_accumulation_steps 20 \
  --eval_steps 200 \
  --objective_mode 0 \
  --max_generations 40 \
  --learning_rate 5e-4 \
  --per_example_max_grad_norm 0.1 \
  --target_epsilon 5 \
  --target_delta 1e-5 \
  --epochs 50
```

## Running evaluation

```bash
python -m gpt2.evaluate_generations --task eval
```

## Playing with decoding

This script creates a command and calls `gpt2/decoding.py`.

```bash
python -m gpt2.launchers.decoding_061821
```
