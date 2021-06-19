# python -m gpt2.launchers.decoding_061821

import os

import fire

from . import shared


def main():
    command = shared._get_command(
        seed=0,
        tuning_mode="prefixtune",
        nonprivate="no",
        script="gpt2.decoding",
        train_dir="/nlp/scr/lxuechen/tests/decoding",
        mid_dim=512,
        preseqlen=10,
        mode="local",
    )
    print(command)
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
