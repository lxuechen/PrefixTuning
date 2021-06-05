from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_WITH_LM_HEAD_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
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
        default=2,
        metadata={
            "help": "In prefixtuning setting, the objective function... "
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
    init_random: Optional[str] = field(
        default='no',
        metadata={
            "help": "whether to init a random embedding, or use GPT2 embedding for the prefix tuning model. "
        },
    )
    use_dropout: Optional[str] = field(
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
    dataless_sample_size: Optional[int] = field(
        default=8,
        metadata={
            "help": "the size of samples for each class in dataless training."
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
    dataless_sample_length: Optional[int] = field(
        default=20,
        metadata={
            "help": "the length of samples for each class in dataless training."
        },
    )
    dataless_control_type: Optional[int] = field(
        default=0,
        metadata={
            "help": "the type of control in dataless training."
        },
    )
    dataless_usebaseline: Optional[str] = field(
        default='yes',
        metadata={
            "help": "use baseline in dataless training."
        },
    )
    dataless_discri_model_path: Optional[str] = field(
        default='textattack/roberta-base-imdb',
        metadata={
            "help": "the path to discri_model and discri_tokenizer"
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

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
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

    dataless: Optional[str] = field(
        default='no', metadata={"help": "Whether we are training or loading dataless model."}
    )

    train_embs: Optional[str] = field(
        default='no', metadata={"help": "whether the train word embeddings"}
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
