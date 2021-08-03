#!/usr/bin/env bash

conda create -n lxuechen-prefix-tuning python=3.8
pip install -r lxuechen_utils_requirements.txt
pip install -r ./gpt2stuff/requirements.txt

# Make sure rust is installed.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install the modified transformers lib.
cd ./transformers/
pip install -e .
cd -
