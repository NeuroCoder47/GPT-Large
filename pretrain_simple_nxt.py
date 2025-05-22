import os
from pathlib import Path
import time
import math  # Needed for cosine decay
import torch
import sentencepiece as spm
from previous_chapters_nxt import *
from utils_t import *
from itertools import cycle
from contextlib import nullcontext
import argparse
import functools
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Distributed and FSDP imports.
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Helper functions.
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data

def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader

def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)

def print_eta(start_time, book_start_time, books_processed, total_books):
    book_end_time = time.time()  # End time of processing the last book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_books - books_processed
    average_time_per_book = total_elapsed_time / books_processed if books_processed > 0 else 0
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")

# Helper function: get_lr modified to include initial_lr.
def get_lr(it, warmup_steps, max_iters, peak_lr, initial_lr, min_lr):
    # Linear warmup.
    if it < warmup_steps:
        return initial_lr + it * ((peak_lr - initial_lr) / warmup_steps)
    # If passed maximum iterations, return min_lr.
    if it > max_iters:
        return min_lr
    # Cosine decay between warmup and max_iters.
    decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

def evaluate_loss(model, data_loader, device, num_batches=5):
    """Evaluate average loss over a few batches from the provided data loader."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):import os
from pathlib import Path
import time
import math  # Needed for cosine decay
import torch
import sentencepiece as spm
from previous_chapters_nxt import * # Assuming these contain necessary definitions like GPTModel, calc_loss_batch, etc.
from utils_t import * # Assuming these contain necessary definitions like create_dataloader_v1, evaluate_model, generate_and_print_sample, plot_losses
from itertools import cycle
from contextlib import nullcontext
import argparse # Kept import as it was in original Code 2, though not used in the final __main__
import functools

from torch.distributed.fsdp import MixedPrecision
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Distributed and FSDP imports.
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Helper functions.
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,       # FP16 for model parameters
    reduce_dtype=torch.float16,      # FP16 for gradient reduction
    buffer_dtype=torch.float16       # FP16 for buffers
)
def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0): # Original signature
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1( # Assuming create_dataloader_v1 is defined elsewhere
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader_v1( # Assuming create_dataloader_v1 is defined elsewhere
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader

def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)

def print_eta(start_time, book_start_time, books_processed, total_books):
    book_end_time = time.time()  # End time of processing the last book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_books - books_processed
    average_time_per_book = total_elapsed_time / books_processed if books_processed > 0 else 0
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")

# Helper function: get_lr modified to include initial_lr.
def get_lr(it, warmup_steps, max_iters, peak_lr, initial_lr, min_lr):
    # Linear warmup.
    if it < warmup_steps:
        return initial_lr + it * ((peak_lr - initial_lr) / warmup_steps)
    # If passed maximum iterations, return min_lr.
    if it > max_iters:
        return min_lr
    # Cosine decay between warmup and max_iters.
    decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

def evaluate_loss(model, data_loader, device, num_batches=5):
    """Evaluate average loss over a few batches from the provided data loader."""
    model.eval()
    total_loss = 0.0
    count = 0
    # ADDED: Scoped autocast for evaluation, using float16 for CUDA
    eval_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            # ADDED: Pinned memory + non-blocking for evaluation data if using CUDA and data_loader isn't handling it (e.g. num_workers=0)
            if device.type == "cuda":
                input_batch = input_batch.pin_memory() # Pin memory
                target_batch = target_batch.pin_memory() # Pin memory
                input_batch, target_batch = input_batch.to(device, non_blocking=True), target_batch.to(device, non_blocking=True)
            else:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # MODIFIED: Apply autocast context
            with eval_ctx:
                loss = calc_loss_batch(input_batch, target_batch, model, device) # Assuming calc_loss_batch is defined elsewhere
            total_loss += loss.item()
            count += 1
            if i + 1 >= num_batches:
                break
    model.train()
    return total_loss / count if count > 0 else float('nan')

# MODIFIED SIGNATURE: Added world_size_for_dist
def train_model_simple_iter(model, optimizer, device, n_iterations,
                            eval_freq, eval_iter, print_sample_iter, start_context,
                            output_dir, save_ckpt_freq, tokenizer,
                            batch_size=1024, train_ratio=0.90, initial_lr=3e-05,
                            all_files=None, total_files=None, GPT_CONFIG_124M=None,
                            grad_accum_steps=1, warmup_steps=1000, min_lr=1e-6,
                            test_loader=None, checkpoint_path=None, world_size_for_dist=1):
    """
    Iteration-based training loop.
    """
    # Initialize tracking variables.
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen = 0
    global_step = 0
    start_time = time.time()
    
    # Get peak learning rate from optimizer.
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = n_iterations
    
    if all_files is None:
        raise ValueError("all_files must be provided with a list of file paths.")
    # Use cycle from the imported itertools.
    file_iterator = iter(all_files) if len(all_files) == 0 else cycle(all_files)
    books_processed = 0

    # Set up mixed precision context.
    # Original ctx was: torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    # This is already good for scoped autocast. Ensured float16 for GradScaler compatibility on CUDA.
    ctx = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    # ADDED: GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


    # Set up test loss logging if a test_loader is provided.
    if test_loader is not None:
        log_file = os.path.join(output_dir, "log.txt")
        with open(log_file, "w") as f:
            f.write("")

    try:
        # Loop until the desired number of iterations is reached.
        while global_step < n_iterations:
            book_start_time = time.time()
            file_path = next(file_iterator)
            books_processed += 1

            text_data = read_text_file(file_path) + " <|endoftext|> "
            print(f"Tokenizing file {books_processed} of ~{total_files}: {file_path}")

            # Create new data loaders for this file.
            train_loader, val_loader = create_dataloaders( # Original call
                text_data,
                train_ratio=train_ratio,
                batch_size=batch_size,
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                num_workers=0 # Original value, implies manual pinning needed for performance
            )
            
            # Create an iterator for training batches.
            train_data_iter = iter(train_loader)
            
            # Process all batches from the current file.
            while True:
                t0 = time.time()  # Start timing for tokens/sec calculation.
                # MODIFIED: optimizer.zero_grad() with set_to_none=True
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                micro_counter = 0
                for i in range(grad_accum_steps):
                    try:
                     input_batch, target_batch = next(train_data_iter)
                     micro_counter += 1
                    except StopIteration:
                     break
                    # ADDED: Pinned memory + non-blocking for training data since num_workers=0 for train_loader
                    if device.type == "cuda":
                        input_batch = input_batch.pin_memory() # Pin memory
                        target_batch = target_batch.pin_memory() # Pin memory
                        input_batch, target_batch = input_batch.to(device, non_blocking=True), target_batch.to(device, non_blocking=True)
                    else:
                        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                    
                    with ctx: # Original scoped autocast
                     loss = calc_loss_batch(input_batch, target_batch, model, device) # Assuming calc_loss_batch is defined
                    loss = loss / grad_accum_steps # Original scaling for grad_accum
                    
                    # MODIFIED: Incorporate scaler with existing no_sync logic, using world_size_for_dist
                    is_distributed_run = world_size_for_dist > 1
                    if is_distributed_run and grad_accum_steps > 1 and i < grad_accum_steps - 1:
                     with model.no_sync(): # Original no_sync logic
                      scaler.scale(loss).backward() # Use scaler
                    else:
                     scaler.scale(loss).backward() # Use scaler
                    accum_loss += loss.item() * grad_accum_steps # Original accum_loss update logic
                    tokens_seen += input_batch.numel()
                # Exit the inner loop if no batches were processed.
                if micro_counter == 0:
                    break

                # Update learning rate by calling get_lr.
                lr = get_lr(global_step, warmup_steps, total_training_steps, peak_lr, initial_lr, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                track_lrs.append(lr)

                # MODIFIED: Proper backward/step order with GradScaler
                # 1. Unscale gradients before clipping (if scaler is enabled)
                if device.type == "cuda": # scaler is only active and enabled on CUDA
                    scaler.unscale_(optimizer)
                # 2. Gradient Clipping (original)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # 3. Optimizer step (with scaler)
                scaler.step(optimizer)
                # 4. Update scaler
                scaler.update()

                global_step += 1

                # (Optional) tokens per second calculation. Kept original calculation
                dt = time.time() - t0
                tokens_processed = input_batch.numel() * grad_accum_steps
                tokens_per_sec = tokens_processed / dt if dt > 0 else 0

                # Evaluation on the current file's train/validation data.
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter) # Assuming evaluate_model defined
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Iteration {global_step}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}") # Original print
                    if print_sample_iter and (global_step % print_sample_iter == 0):
                        generate_and_print_sample(model, tokenizer, device, start_context) # Assuming defined
                
                # Log test loss every 50 iterations if test_loader is provided.
                if test_loader is not None and global_step % 50 == 0:
                    test_loss = evaluate_loss(model, test_loader, device) # Uses modified evaluate_loss
                    log_line = f"step {global_step:5d} | test loss: {test_loss:.6f} | train loss: {accum_loss:.6f}" # Original print
                    print(log_line)
                    with open(os.path.join(output_dir, "log.txt"), "a") as f:
                        f.write(log_line + "\n")
                
                # Save checkpoint every save_ckpt_freq iterations if a checkpoint path is provided.
                if checkpoint_path and (global_step % save_ckpt_freq == 0): # Original conditional
                    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    checkpoint = {
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "config": getattr(model, "config", None),
                        # ADDED: Save scaler state if on CUDA
                        "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else None
                    }
                    ckpt_file = os.path.join(output_dir, f"model_pg_{global_step}.pth")
                    torch.save(checkpoint, ckpt_file)
                    print(f"Checkpoint saved to {ckpt_file}")
                
                if global_step >= n_iterations:
                    break
            
            if books_processed <= total_files:
                print_eta(start_time, book_start_time, books_processed, total_files)
                
    except KeyboardInterrupt:
        ckpt_file = os.path.join(output_dir, f"model_pg_{global_step}_interrupted.pth")
        # MODIFIED: Original save was model.state_dict() directly, now saving checkpoint dict including scaler
        model_state_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        interrupted_checkpoint = {
            "model_state_dict": model_state_to_save,
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "config": getattr(model, "config", None),
            # ADDED: Save scaler state if on CUDA
            "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else None
        }
        torch.save(interrupted_checkpoint, ckpt_file)
        print(f"Saved {ckpt_file} due to keyboard interrupt.")
    
    return train_losses, val_losses, track_tokens_seen


# MODIFIED SIGNATURE: Added compile_model_flag
def run_training(
    data_dir: str = "dataset_output_min_change_fixed_dl/chunks",
    output_dir: str = "model_checkpoints",
    n_iterations: int = 10000,
    print_sample_iter: int = 1000,
    eval_freq: int = 100,
    save_ckpt_freq: int = 100_000,
    lr: float = 5e-4,
    batch_size: int = 32,
    start_context: str = "Every effort moves you",
    compile_model_flag: bool = True, # ADDED: Flag for torch.compile with default
):
    # Initialize distributed training if applicable.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Initializing distributed training (rank {rank}/{world_size})...")
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1

    # Set device using LOCAL_RANK for distributed runs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    
    # ADDED: TF32 settings for CUDA
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Your model configuration.
    GPT_CONFIG_124M_n = {
    "vocab_size": 131072,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
    }
   
            
    
    # Instantiate and move model to device.
    model = GPTModel(GPT_CONFIG_124M_n) # Assuming GPTModel defined elsewhere
    model.to(device)
    model = torch.compile(model)
    if world_size > 1:
        print(f"Wrapping model with FSDP for {world_size} processes.")
        my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=20000)
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision= mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=False),
            # ADDED: device_id for FSDP, critical for CUDA operations within FSDP
            device_id=torch.cuda.current_device() if device.type == "cuda" else None
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1) # Original optimizer setup
    
    # Initialize tokenizer.
    tokenizer = spm.SentencePieceProcessor(model_file='test_bpe.model')
    
    # Gather training files.
    all_files = [
        os.path.join(path, name)
        for path, subdirs, files in os.walk(data_dir)
        for name in files if name.endswith((".txt"))
    ]
    total_files = len(all_files)
    if total_files == 0:
        print("No training text files found. Make sure you selected the correct input directory")
        return
    print("Total files:", total_files)
    
    # Create the output directory.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # MODIFIED CALL: Pass world_size to train_model_simple_iter
    train_losses, val_losses, tokens_seen = train_model_simple_iter(
        model, optimizer, device,
        n_iterations=n_iterations,
        eval_freq=eval_freq,
        eval_iter=1, # Original eval_iter value
        print_sample_iter=print_sample_iter,
        output_dir=str(output_path),
        save_ckpt_freq=save_ckpt_freq,
        start_context=start_context,
        tokenizer=tokenizer,
        batch_size=batch_size,
        all_files=all_files,
        total_files=total_files,
        GPT_CONFIG_124M=GPT_CONFIG_124M_n,
        # Default arguments from train_model_simple_iter definition will be used for:
        # grad_accum_steps, warmup_steps, min_lr, test_loader, checkpoint_path
        world_size_for_dist=world_size # Pass the world_size
    )
    
    # Plot losses and save final model.
    if train_losses: # Ensure train_losses is not empty before plotting
        epochs_tensor = torch.linspace(0, n_iterations, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_path) # Assuming plot_losses defined
    final_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(final_state, output_path / "model_pg_final.pth")
    if device.type == "cuda": # Original conditional print
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    run_training() # Original __main__ block




#Terminal Command to Run the Script

#Assuming your script is saved as train_fsdp.py, you can launch it using PyTorch’s distributed launcher (recommended for FSDP):

#torchrun --nproc_per_node=NUM_GPUS train_fsdp.py

#Replace NUM_GPUS with the number of GPUs you want to use. For example, if you have 4 GPUs:

#torchrun --nproc_per_node=4 train_fsdp.py