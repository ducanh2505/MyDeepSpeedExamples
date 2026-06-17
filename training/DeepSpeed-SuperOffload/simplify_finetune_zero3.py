import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed import comm as dist
import torch
import time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    enable_full_determinism, default_data_collator
)
import argparse
import os
from utils import (
    setup_logger,
    load_and_preprocess_dataset,
    get_parameter_count, estimate_transformer_tflops
)
from .constants import (
    DEFAULT_OPTIMIZER_LR, DEFAULT_OPTIMIZER_BETAS, MS_PER_SECOND
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune language models with DeepSpeed ZeRO Stage 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name or path")
    parser.add_argument("--lr", type=float, required=True,
                       help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, required=True,
                       help="Training batch size per device")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save model checkpoints")
    
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                       choices=["eager", "sdpa", "flash_attention_2"],
                       help="Attention implementation to use")
    parser.add_argument("--leaf_module", type=str, default=None,
                        help="Set leaf_module to enable fine-tuning MoE models")
    parser.add_argument("--activation_checkpointing", action="store_true",
                       help="Enable activation checkpointing to save memory")

    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimization")
    parser.add_argument("--warmup", type=float, default=0.01,
                       help="Warmup ratio for learning rate schedule")
    
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank passed from distributed launcher")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true",
                       help="Enable deterministic training for full reproducibility")
    
    parser.add_argument("--log_interval", type=int, default=1,
                       help="Log performance metrics every N steps")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level for controlling output verbosity")
    parser.add_argument("--warmup_steps", type=int, default=15,
                       help="Number of warmup steps for performance measurements")
    parser.add_argument("--bench_steps", type=int, default=100,
                       help="Number of benchmark steps to run")
    
    
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset_percentage", type=float, default=100.0,
                       help="Percentage of dataset to use (1.0-100.0)")
    
    return parser

parser = create_argument_parser()
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
model_name = args.model_name
attn_implementation = args.attn_implementation

enable_full_determinism(args.seed)
torch.backends.cudnn.benchmark = False
logger = setup_logger(rank=0, log_level=args.log_level)
logger.debug("Training configuration:")
logger.debug(f"  Model: {args.model_name}")
logger.debug(f"  Batch size: {args.batch_size}")
logger.debug(f"  Max length: {args.max_length}")
logger.debug(f"  Learning rate: {args.lr}")
logger.debug(f"  Epochs: {args.num_train_epochs}")
logger.debug(f"  Activation checkpointing: {args.activation_checkpointing}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.debug(f"Set pad_token to eos_token: {tokenizer.eos_token}")
logger.debug(f"Loading model: {model_name}")
logger.debug(f"Attention implementation: {attn_implementation}")

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation
)
logger.debug("Enabling gradient checkpointing...")
if hasattr(model.config, 'use_cache'):
    model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

    
optimizer = DeepSpeedCPUAdam(
    model.parameters(), 
    lr=DEFAULT_OPTIMIZER_LR, 
    betas=DEFAULT_OPTIMIZER_BETAS
)
tokenized_dataset, train_dataloader = load_and_preprocess_dataset(
    args.dataset_name, args.dataset_percentage, tokenizer, args.max_length, logger
)

# Initialize DeepSpeed
model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
    training_data=tokenized_dataset,
    collate_fn=default_data_collator
)
logger = setup_logger(rank=dist.get_rank(), log_level=args.log_level)

model_engine.train()
sequence_length = args.max_length
model_size = sum(get_parameter_count(p) for p in model.parameters())
logger.debug(f"Model size: {model_size:,} parameters")

total_tflops = None
total_tflops = estimate_transformer_tflops(
            sequence_length, model_size, model.config.num_hidden_layers, 
            model.config.hidden_size, args.activation_checkpointing
        )

global_step = 0
total_tokens_processed = 0
total_train_time = 0
iter_times = []
losses = []

stop = False

for epoch in range(args.num_train_epochs):
    logger.debug(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")
    
    for step, batch in enumerate(train_dataloader):
        step_start_time = time.time()
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}
        
        actual_batch_size = batch['input_ids'].shape[0]
        tokens_in_batch = actual_batch_size * sequence_length
        
        outputs = model_engine(**batch)
        loss = outputs.loss

        model_engine.backward(loss)
        
        model_engine.step()

        step_time = time.time() - step_start_time
        global_step += 1
        
        if global_step > args.warmup_steps:
            iter_times.append(step_time)
        
        losses.append(loss.item())
        
        total_tokens_processed += tokens_in_batch
        total_train_time += step_time
        
        tokens_per_second = tokens_in_batch / step_time
        step_tflops = None
        
        if total_tflops is not None:
            step_tflops = args.batch_size * total_tflops / step_time
        
        if global_step % args.log_interval == 0:
            avg_loss = sum(losses[-args.log_interval:]) / len(losses[-args.log_interval:])
            log_msg = (f"Step {global_step:4d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Time: {step_time * MS_PER_SECOND:5.0f}ms | "
                        f"TFLOPS: {step_tflops:5.2f} | "
                        f"Tokens/s: {tokens_per_second:6.0f}")

            logger.info(log_msg)
        stop = global_step >= args.bench_steps
        if stop:
            break
    
    if stop:
        break
            

if dist.get_rank() == 0:
    try:
        logger.debug(f"Saving model to {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)
        model_engine.save_checkpoint(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.debug("Model saved successfully!")
    except Exception as e:
        logger.error(f"Error saving model: {e}")