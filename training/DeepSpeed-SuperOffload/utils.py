import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator
)
from datasets import load_dataset
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

import logging
from constants import (
    LOGGER_LEVEL, LOGGER_NAME, LOGGER_RANK, LOGGER_FMT, LOGGER_DATE_FMT,
    TFLOPS_DENOMINATOR, DEFAULT_OPTIMIZER_LR, DEFAULT_OPTIMIZER_BETAS,
    ALPACA_INSTRUCTION_TEMPLATE, ALPACA_INPUT_TEMPLATE, ALPACA_RESPONSE_TEMPLATE
    
)
def setup_logger(rank: int = 0, log_level: str = LOGGER_LEVEL) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    if rank == LOGGER_RANK:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt=LOGGER_FMT,
            datefmt=LOGGER_DATE_FMT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_parameter_count(parameter: torch.nn.Parameter) -> int:
    return parameter.ds_numel if hasattr(parameter, "ds_tensor") else parameter.numel()


def estimate_transformer_tflops(
    seq_len: int, 
    model_size: int, 
    num_layers: int, 
    hidden_size: int, 
    use_activation_checkpointing: bool = False
) -> float:
    """
    Estimate TFLOPS for decoder-only densde models.
    """
    coefficient = 4 if use_activation_checkpointing else 3
    tflops = (
        2 * coefficient * model_size * seq_len
        + 2 * 2 * coefficient * num_layers * hidden_size * seq_len**2
    ) / TFLOPS_DENOMINATOR
    return tflops

def preprocess_alpaca_example(
    example: Dict[str, str], 
    tokenizer: AutoTokenizer, 
    max_length: int = 2048
) -> Dict[str, Any]:
    prompt = ALPACA_INSTRUCTION_TEMPLATE.format(instruction=example['instruction'])
    
    if example.get("input", "").strip():
        prompt += ALPACA_INPUT_TEMPLATE.format(input=example['input'])
    
    prompt += ALPACA_RESPONSE_TEMPLATE.format(output=example['output'])
    
    tokenized = tokenizer(
        prompt, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length",
        return_tensors=None
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def detect_moe_model(model: AutoModelForCausalLM, model_name: str) -> bool:
    moe_config_attrs = [
        'num_local_experts', 'moe_layers', 'num_experts', 
        'expert_capacity', 'router_aux_loss_coef'
    ]
    
    for attr in moe_config_attrs:
        if hasattr(model.config, attr):
            return True
    return False


def create_experiment_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = args.model_name.split("/")[-1]
    activation_checkpointing = 1 if args.activation_checkpointing else 0

    exp_name = (f"{model_name_short}_bs{args.batch_size}_seq{args.max_length}"
                f"_ac{activation_checkpointing}_T{timestamp}")
    return exp_name

def load_tokenizer(model_name: str, logger: logging.Logger) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    return tokenizer


def load_model(model_name: str, attn_implementation: str, logger: logging.Logger) -> AutoModelForCausalLM:
    logger.debug(f"Loading model: {model_name}")
    logger.debug(f"Attention implementation: {attn_implementation}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation
    )
    
    return model


def setup_model_training(model: torch.nn.Module, use_activation_checkpointing: bool = True, logger: logging.Logger = None) -> None:
    if use_activation_checkpointing:
        if logger:
            logger.debug("Enabling gradient checkpointing...")
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def create_optimizer(model: AutoModelForCausalLM) -> Any:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    optimizer = DeepSpeedCPUAdam(
        model.parameters(), 
        lr=DEFAULT_OPTIMIZER_LR, 
        betas=DEFAULT_OPTIMIZER_BETAS
    )
    return optimizer


def load_and_preprocess_dataset(
    dataset_name: str, 
    dataset_percentage: float, 
    tokenizer: AutoTokenizer, 
    max_length: int,
    logger: logging.Logger
) -> Tuple[Any, DataLoader]:
    logger.debug(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name)
    original_size = len(dataset["train"])
    
    if dataset_percentage < 100.0:
        subset_size = int(original_size * dataset_percentage / 100.0)
        dataset["train"] = dataset["train"].select(range(subset_size))
        logger.debug(f"Using {dataset_percentage}% of dataset: {subset_size}/{original_size} examples")
    else:
        logger.debug(f"Using full dataset: {original_size} examples")

    logger.debug("Tokenizing dataset...")
    
    tokenized_dataset = dataset["train"].map(
        lambda x: preprocess_alpaca_example(x, tokenizer, max_length), 
        batched=False,
        desc="Tokenizing"
    )
    
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=True
    )
    
    return tokenized_dataset, train_dataloader
