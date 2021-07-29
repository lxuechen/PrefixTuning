#!/bin/bash

for seq_len in 32 64 128 256 512 1024; do
  python -m memory.batch_size --out_dir "/nlp/scr/lxuechen/prefixtune/memory" --seq_len ${seq_len}
done
