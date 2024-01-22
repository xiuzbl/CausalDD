import deepspeed
from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments, HfArgumentParser


@dataclass
class MyArguments:
    dataid: str = field(default=None)
    wiki_disturb_train_file: str = field(default=None) # for TIE disturb, unlikelihood training
    unlikelihood: bool = field(default=False) # for TIE disturb
    reddit_wo_evidence_train_file: str = field(default=None)
    wiki_wo_evidence_train_file: str = field(default=None)
    add_reddit_wo_evidence: bool = field(default=False)
    add_wiki_wo_evidence: bool = field(default=False)
    add_disturb_lm: bool = field(default=False)
    add_disturb_kd: bool = field(default=False)
    alpha_kd: float = field(default=1.0)
    KD_temperature: float = field(default=2.0)
    disturb: bool = field(default=False)
    contrast: bool = field(default=False)
    nli_threshold: float = field(default=0.5)
    nli_weighted: bool = field(default=False)
    nli_selected: bool = field(default=False)
    add_uttr_generating: bool = field(default=False)
    add_understanding: bool = field(default=False)
    add_responding: bool = field(default=False)
    add_grounding: bool = field(default=False)
    mix_twotasks: bool = field(default=False)
    add_wikidata: bool = field(default=False)
    add_redditdata: bool = field(default=False)
    num_data: int = field(default=1000000)
    wikidialog_file: str = field(default='./data/wikidialog.json')
    wikidialog_valid_file: str = field(default='./data/wikidialog.json')
    use_pseudo_data: bool = field(default=False)
    wiki_pseudo_train_file: str = field(default='')
    reddit_pseudo_train_file: str = field(default='')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default='./PRETRAIN/cache/t5-base'

    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    from_checkpoint: bool = field(
        default=False, metadata={"help": "Whether load from checkpoint to continue learning"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    # fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(default=None)
    n_gpu: int = field(init=False, repr=False, default=-1)
    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
                    "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default='text',
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    record_column: Optional[str] = field(
        default='record',
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocess: bool = field(
        default=True,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessed_folder: Optional[str] = field(
        default=None,
        metadata={
            "help": "Folder to preprocessed data"
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_prefix_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum prefix length."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    meta_negative: int = field(
        default=-1, metadata={"help": "Negative Schema Number in Training."}
    )
    ordered_prompt: bool = field(
        default=True,
        metadata={
            "help": "Whether to sort the spot prompt and asoc prompt or not."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

    decoding_format: str = field(
        default='spotasoc',
        metadata={"help": "Decoding Format"}
    )
    record_schema: str = field(
        default=None, metadata={"help": "The input event schema file."}
    )
    spot_noise: float = field(
        default=0., metadata={"help": "The noise rate of null spot."}
    )
    asoc_noise: float = field(
        default=0., metadata={"help": "The noise rate of null asoc."}
    )
    meta_positive_rate: float = field(
        default=1., metadata={"help": "The keep rate of positive spot."}
    )

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, MyArguments))
# parser = deepspeed.add_config_arguments(parser)
model_args, data_args, training_args, myargs = parser.parse_args_into_dataclasses()
