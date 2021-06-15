from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoConfig, GPT2LMHeadModel

from gpt2 import prefix_tuning_minimal


@dataclass
class ModelArgs:
    preseqlen: int = field(default=5)
    mid_dim: int = field(default=512)
    model_name_or_path: str = field(default="gpt2-medium")
    cache_dir: Optional[str] = field(default=None)
    prefix_dropout: float = field(default=0)


def test_gpt2_freezes():
    model_args = ModelArgs()
    config = AutoConfig.from_pretrained("gpt2-medium")
    model = prefix_tuning_minimal.PrefixTuningMinimal(config=config, model_args=model_args)
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=config)

    p1 = torch.cat([p.view(-1) for p in model.gpt2.parameters()])
    p2 = torch.cat([p.view(-1) for p in gpt2.parameters()])
    torch.testing.assert_allclose(p1, p2)


def test_attend_to_past_key_values():
    model_args = ModelArgs()
    config = AutoConfig.from_pretrained("gpt2-medium")
    model = prefix_tuning_minimal.PrefixTuningMinimal(config=config, model_args=model_args)
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=config)

    print(model.training)
    print(gpt2.training)

    model.eval()
    gpt2.eval()

    inputs_ids = torch.tensor([[10, 293, 3934, 97]])
    output_ids = model.generate(input_ids=inputs_ids)
    print(output_ids)

    output_ids = model.gpt2.generate(input_ids=inputs_ids)
    print(output_ids)


if __name__ == "__main__":
    test_attend_to_past_key_values()
