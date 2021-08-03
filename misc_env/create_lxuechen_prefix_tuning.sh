#!/usr/bin/env bash

conda create -n lxuechen-prefix-tuning python=3.8 -y
sleep 3
conda init bash
conda activate lxuechen-prefix-tuning
exit

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r lxuechen_utils_requirements.txt
pip install -r ./gpt2stuff/requirements.txt

# Make sure rust is installed.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install the modified transformers lib.
cd ./transformers/
pip install -e .
cd -
