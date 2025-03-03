# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import logging

import torch
import transformers
import argparse
from transformers import set_seed

from transformers import AutoTokenizer

from lora_utils import (
    SavePeftModelCallback,
    print_trainable_parameters,
    get_last_checkpoint,
    DEFAULT_PAD_TOKEN,
)
from data_utils.data_utils_rm import make_binary_reward_modeling_data_module
from models.reward_model import (
    RewardConfig,
    RewardModel,
    RewardModelTrainer as Trainer,
    compute_reward_modeling_metrics,
)

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.train.train import smart_tokenizer_and_embedding_resize
from data_utils.common_utils import preprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_default_dtype(torch.bfloat16)

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(
        default=500,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # From LLaVA
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    # From AlpacaFarm
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice", "nrmse_0", "nrmse_1"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learning rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )
    
    # FSDP-specific arguments
    fsdp: Optional[str] = field(
        default=None,
        metadata={
            "help": "Enable FSDP training. Choose from 'full_shard', 'shard_grad_op', 'hybrid_shard', or 'offload'",
        },
    )
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of transformer layer class names to wrap with FSDP",
        },
    )
    fsdp_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a JSON file containing FSDP configuration",
        },
    )
    fsdp_min_num_params: int = field(
        default=0,
        metadata={
            "help": "FSDP's minimum number of parameters for Default Auto Wrapping",
        },
    )
    # Advanced FSDP configuration
    fsdp_backward_prefetch: Optional[str] = field(
        default="backward_pre", 
        metadata={"help": "FSDP backward prefetch style: 'backward_pre' or 'backward_post'"}
    )
    fsdp_forward_prefetch: Optional[bool] = field(
        default=False, 
        metadata={"help": "Enable FSDP forward prefetch"}
    )
    fsdp_cpu_offload: bool = field(
        default=False, 
        metadata={"help": "Enable CPU offloading in FSDP"}
    )
    fsdp_state_dict_type: Optional[str] = field(
        default="sharded",
        metadata={"help": "FSDP state dict type: 'full', 'sharded', or 'local'"}
    )
    fsdp_activation_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable activation checkpointing in FSDP"}
    )
    fsdp_use_orig_params: bool = field(
        default=True,
        metadata={"help": "Use original parameters in FSDP, needed for gradient checkpointing"}
    )
    fsdp_sync_module_states: bool = field(
        default=True,
        metadata={"help": "Sync module states at the beginning of training"}
    )


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # Initialize FSDP config if using FSDP
    fsdp_config = None
    if args.fsdp:
        try:
            # Check if fsdp_config is already a dict (parsed by HuggingFace)
            if hasattr(args, 'fsdp_config') and isinstance(args.fsdp_config, dict):
                fsdp_config = args.fsdp_config
            # Check if it's a string (path to a file)
            elif hasattr(args, 'fsdp_config') and isinstance(args.fsdp_config, str):
                with open(args.fsdp_config, "r") as f:
                    fsdp_config = json.load(f)
            else:
                # No valid config provided, create default
                fsdp_config = {}
        except Exception as e:
            # If any error occurs, create a default config
            print(f"Warning: Error loading FSDP config: {e}")
            fsdp_config = {}
        
        # Setup FSDP configuration
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            BackwardPrefetch,
            ShardingStrategy,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        # Get the right sharding strategy as STRING
        if args.fsdp == "full_shard":
            sharding_strategy = "FULL_SHARD"
        elif args.fsdp == "shard_grad_op":
            sharding_strategy = "SHARD_GRAD_OP"
        elif args.fsdp == "hybrid_shard":
            sharding_strategy = "HYBRID_SHARD"
        else:
            sharding_strategy = "FULL_SHARD"
        
        # Set up backward prefetch as STRING
        backward_prefetch = args.fsdp_backward_prefetch  # Use the string directly
        
        # Setup CPU offload
        if args.fsdp_cpu_offload:
            cpu_offload = {"offload_params": True}
        else:
            cpu_offload = {"offload_params": False}
        
        # Setup mixed precision as dict of strings
        if args.bf16:
            mixed_precision = {
                "param_dtype": "bfloat16",
                "reduce_dtype": "bfloat16",
                "buffer_dtype": "bfloat16",
            }
        elif args.fp16:
            mixed_precision = {
                "param_dtype": "float16",
                "reduce_dtype": "float16",
                "buffer_dtype": "float16",
            }
        else:
            mixed_precision = None
        
        # Configure wrapping policy
        auto_wrap_policy = None
        if hasattr(args, 'fsdp_transformer_layer_cls_to_wrap') and args.fsdp_transformer_layer_cls_to_wrap:
            # We'll handle the actual wrapping policy later in Trainer
            # Just record the name of the transformer layer class
            fsdp_transformer_layer_cls_to_wrap = args.fsdp_transformer_layer_cls_to_wrap
        
        # Create final FSDP config
        fsdp_config.update({
            "sharding_strategy": sharding_strategy,
            "backward_prefetch": backward_prefetch,
            "forward_prefetch": args.fsdp_forward_prefetch,
            "cpu_offload": cpu_offload,
            "mixed_precision": mixed_precision,
            "fsdp_transformer_layer_cls_to_wrap": args.fsdp_transformer_layer_cls_to_wrap if hasattr(args, 'fsdp_transformer_layer_cls_to_wrap') else None,
            "xla": False,
            "sync_module_states": args.fsdp_sync_module_states,
            "use_orig_params": args.fsdp_use_orig_params,
        })
        
        # Set state dict type for saving
        if args.fsdp_state_dict_type == "full":
            fsdp_config["state_dict_type"] = "full"
        elif args.fsdp_state_dict_type == "sharded":
            fsdp_config["state_dict_type"] = "sharded"
        else:
            fsdp_config["state_dict_type"] = "sharded"  # default
            
    # Apply FSDP config to training args
    if fsdp_config:
        training_args.fsdp_config = fsdp_config

    # Rest of the function remains unchanged
    if args.resume_dir is not None:
        checkpoint_dir, completed_training = args.resume_dir, False
    else:
        checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    if completed_training:
        rank0_print("Detected that training was already completed!")

    if checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)
        if args.resume_from_training:
            rank0_print("Resuming from training not supported yet. Exiting.")
            exit(1)

    tokenizer_model_name = args.model_name_or_path
    TokenizerClass = AutoTokenizer

    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    if model_args.vision_tower is not None:
        from llava.model import LlavaLlamaForCausalLM

        # For FSDP, we need to delay creating the model until after
        # the FSDP wrapper configuration is complete
        with DisableLogger():
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16,
                quantization_config=None,
                low_cpu_mem_usage=True, 
                trust_remote_code=True,
                device_map={"": torch.cuda.current_device()} if not args.fsdp else None,
            )

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        data_args.image_processor = vision_tower.image_processor
        
            
        data_args.is_multimodal = True
        model_args.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end

    data_module = make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    if args.do_train:
        training_data = data_module["train_dataset"]
        rank0_print("Training data size:", len(training_data))
        rank0_print("Training data example:")
        for i in range(min(3, len(training_data))):
            ex_input_ids_0 = training_data[i]["input_ids"][0]
            ex_input_ids_0[ex_input_ids_0 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
            rank0_print(tokenizer.decode(ex_input_ids_0, skip_special_tokens=False))
            rank0_print("=" * 20)
            ex_input_ids_1 = training_data[i]["input_ids"][1]
            ex_input_ids_1[ex_input_ids_1 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
            rank0_print(
                tokenizer.decode(
                    ex_input_ids_1,
                    skip_special_tokens=False,
                )
            )
            rank0_print("=" * 20)
            rank0_print("=" * 20)

    config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

    with DisableLogger():
        model = RewardModel(
            args=args,
            config=config,
            qlora=True,
            checkpoint_dir=checkpoint_dir,
            tokenizer=tokenizer,
        )

    model.backbone_model.config.use_cache = False
    print_trainable_parameters(args, model)
    print("loaded model")
    set_seed(args.seed)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if args.do_train or args.do_eval:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
