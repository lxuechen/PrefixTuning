from dataclasses import dataclass, field
import sys
from typing import Optional

import transformers

MODEL_CONFIG_CLASSES = list(transformers.MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from "
                    "scratch."
        },
    )
    prefixModel_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The prefix model checkpoint for weights initialization. "
                    "Leave None if you want to train a model from scratch."
        },
    )

    prefix_mode: Optional[str] = field(
        default='activation',
        metadata={
            "help": "activation or embedding"
        },
    )

    preseqlen: Optional[int] = field(
        default=0,
        metadata={
            "help": "preseqlen for how many tokens of prefix should we include."
        },
    )

    optim_prefix: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether we are optimizing the prefix directly, or optimize another amortized function that "
                    "genrate the prefix."
        },
    )

    tuning_mode: Optional[str] = field(
        default='finetune',
        metadata={
            "help": "whether it's doing prefixtune or finetune."
        },
    )

    objective_mode: Optional[int] = field(
        default=1,
        metadata={
            "help": "0 is the usual token-level objective (not suitable for DP);"
                    "1 is the line-level objective (suitable for DP)"
        },
    )

    top_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": "In finetuning setting, if we only tune the top k layers. "
        },
    )

    adapter_design: Optional[int] = field(
        default=2,
        metadata={
            "help": "For Baseline of the adapter module... (1) means using the NLG adapter reference. "
                    "(2) means using a design similar to adapter module"
        },
    )

    adapter_bottleneck: Optional[int] = field(
        default=100,
        metadata={
            "help": "For baseline adapter module: the mid dim of the adapter. "
        },
    )

    parametrize_emb: Optional[str] = field(
        default='MLP',
        metadata={
            "help": "MLP or Emb to parametrize when we optimize for the embeddings."
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "dropout rate for the prefix tuning model. "
        },
    )

    teacher_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "dropout rate for the teacher model. "
        },
    )

    init_random: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to init a random embedding, or use GPT2 embedding for the prefix tuning model. "
        },
    )

    init_shallow: Optional[str] = field(
        default='no',
        metadata={
            "help": "shallow is default to be no, because we add reparametrization trick. If shallow=yes, "
                    "then no reparametrization "
        },
    )

    init_shallow_word: Optional[str] = field(
        default='no',
        metadata={
            "help": "when init_shallow is yes, what word to use... "
        },
    )

    use_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    use_custom_teacher_dropout: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to use dropout of GPT2 on trainer. "
        },
    )

    mid_dim: Optional[int] = field(
        default=512,
        metadata={
            "help": "the mid dim."
        },
    )

    gumbel: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the gumbel softmax trick in training."
        },
    )

    replay_buffer: Optional[str] = field(
        default='no',
        metadata={
            "help": "use the replay buffer in training."
        },
    )

    training_obj: Optional[int] = field(
        default=0,
        metadata={
            "help": "use a specified training objective"
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(default=None)
    val_data_file: Optional[str] = field(default=None)
    eval_data_file: Optional[str] = field(default=None)

    train_prompt_file: Optional[str] = field(default=None)
    val_prompt_file: Optional[str] = field(default=None)
    eval_prompt_file: Optional[str] = field(default=None)

    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation "
                    "language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The task mode"}
    )

    matching_objective: Optional[str] = field(
        default='kl', metadata={"help": "The distillation objective"}
    )

    distill: Optional[str] = field(
        default='no', metadata={"help": "yes/no"}
    )

    finetuned_model_path: Optional[str] = field(
        default="/u/scr/xlisali/contrast_LM/transformers/examples/full/full/webnlgfinetune_n_20_act_cat_b=6-e"
                "=10_d=0.0_u=no_lr=1e-05_w=0.0_s=101_r=n_m=512_earlystop",
        metadata={"help": "finetuned model path (teacher model)"}
    )

    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )

    lowdata_token: Optional[str] = field(
        default='summarize', metadata={"help": "The token to be prepended at initialization time. "}
    )

    use_lowdata_token: Optional[str] = field(
        default='yes',
        metadata={"help": "Whether we should use the lowdata token and pass it to the prefixTuning Model "
                          "for the initialization trick.  "}
    )

    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )

    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )

    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special "
                    "tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    train_embs: str = field(default="yes")

    max_seq_len: int = field(default=sys.maxsize)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_eval_batches: int = field(default=-1, metadata={"help": "Maximum number of evaluation steps to run."})
    max_generations: int = field(default=sys.maxsize)
    max_generations_train: int = field(default=60)
    ema_model_averaging: str = field(default="yes")
    ema_model_gamma: float = field(default=0.99)
    ema_model_start_from: int = field(default=1000)
    efficient: str = field(default="no")  # Whether to turn on memory-efficient per-sample clipping.
    debug: str = field(default="no")

    def __post_init__(self):
        self.ema_model_averaging = (self.ema_model_averaging.lower() in ('y', 'yes'))
        self.efficient = (self.efficient.lower() in ('y', 'yes'))
        self.debug = (self.debug.lower() in ('y', 'yes'))


@dataclass
class PrivacyArguments:
    """Arguments for differentially private training."""

    per_example_max_grad_norm: float = field(
        default=1., metadata={
            "help": "Clipping 2-norm of per-sample gradients."
        }
    )
    noise_multiplier: float = field(
        default=None, metadata={
            "help": "Standard deviation of noise added for privacy; if `target_epsilon` is specified, "
                    "use the one searched based budget"
        }
    )
    target_epsilon: float = field(
        default=None, metadata={
            "help": "Privacy budget; if `None` use the noise multiplier specified."
        }
    )
    target_delta: float = field(
        default=None, metadata={
            "help": "Lax probability in approximate differential privacy; if `None` use 1 / len(train_data)."
        }
    )
    nonprivate: str = field(
        default="no", metadata={"help": "Train non-privately if True."}
    )
    accounting_mode: str = field(
        default="rdp_cks", metadata={"help": "One of (`rdp`, `gdp`, `rdp_cks`, `all`)."}
    )

    def __post_init__(self):
        if self.target_epsilon < 0:
            self.target_epsilon = None
        if self.target_delta < 0:
            self.target_delta = None
