###################################################################################################
#
# Bi-RWKV 430M - From-Scratch Pre-training Script
#
# This script is designed for the first phase: Masked Language Model (MLM) pre-training.
# It is optimized for multi-GPU training on hardware like 2x 16GB GPUs using
# Hugging Face Accelerate and DeepSpeed ZeRO Stage 2.
#
# Author: Gemini
# Date: 2025-07-13
#
###################################################################################################

import os
import math
import types
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Attempt to import dependencies
try:
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import set_seed
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from transformers import DataCollatorForLanguageModeling, get_scheduler
    import bitsandbytes as bnb
    import triton
    import triton.language as tl
    from tqdm import tqdm
except ImportError as e:
    print(f"Dependency not found: {e.name}. Please install all required packages:")
    print("pip install torch accelerate datasets tokenizers transformers bitsandbytes triton tqdm")
    exit(1)

# --- 1. Configuration ---
# All hyperparameters are managed here.

@dataclass
class TrainingConfig:
    # Model Config
    n_layer: int = 32
    n_embd: int = 1024
    vocab_size: int = 50304 # Standard for GPT-2/RoBERTa tokenizers
    head_size: int = 64 # n_embd must be divisible by head_size
    
    # Training Config
    output_dir: str = "birwkv-430m-pretrained"
    num_train_epochs: int = 1
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 2000
    weight_decay: float = 0.1
    
    # Data Config
    dataset_name: str = "wikitext" # Use "c4" or "the_pile" for large-scale pre-training
    dataset_config_name: str = "wikitext-103-raw-v1"
    tokenizer_path: str = "roberta-base-tokenizer.json" # Download from HF Hub or train your own
    sequence_length: int = 1024
    mask_token_id: int = 50264 # For RoBERTa tokenizer
    mlm_probability: float = 0.15

    # Hardware & DeepSpeed Config
    per_device_train_batch_size: int = 4 # Adjust based on VRAM
    gradient_accumulation_steps: int = 8 # Effective batch size = 2 GPUs * 4 * 8 = 64
    mixed_precision: str = "fp16" # "fp16" or "bf16"
    seed: int = 42

config = TrainingConfig()

# --- 2. Triton Kernel for Bidirectional WKV ---
# This is the core computational engine of our model.

@triton.jit
def bi_wkv_forward_kernel(
    R, K, V, W, U, Y,
    stride_r_b, stride_r_h, stride_r_t, stride_r_n,
    stride_k_b, stride_k_h, stride_k_t, stride_k_n,
    stride_v_b, stride_v_h, stride_v_t, stride_v_n,
    stride_w_h, stride_w_n,
    stride_u_h, stride_u_n,
    stride_y_b, stride_y_h, stride_y_t, stride_y_n,
    T: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    r_ptr = R + b_idx * stride_r_b + h_idx * stride_r_h
    k_ptr = K + b_idx * stride_k_b + h_idx * stride_k_h
    v_ptr = V + b_idx * stride_v_b + h_idx * stride_v_h
    w_ptr = W + h_idx * stride_w_h
    u_ptr = U + h_idx * stride_u_h
    y_ptr = Y + b_idx * stride_y_b + h_idx * stride_y_h
    s_kv_fwd, s_k_fwd, s_kv_bwd, s_k_bwd = (tl.zeros([BLOCK_SIZE], dtype=tl.float32) for _ in range(4))
    n_offsets = tl.arange(0, BLOCK_SIZE)

    for t in range(T - 1, -1, -1):
        k = tl.load(k_ptr + t * stride_k_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + t * stride_v_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        e_k, decay = tl.exp(k), tl.exp(-tl.exp(w))
        s_kv_bwd = s_kv_bwd * decay + e_k * v
        s_k_bwd = s_k_bwd * decay + e_k

    for t in range(0, T):
        k = tl.load(k_ptr + t * stride_k_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + t * stride_v_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + t * stride_r_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        e_k, e_u, decay = tl.exp(k), tl.exp(u), tl.exp(-tl.exp(w))
        
        s_kv_bwd, s_k_bwd = (s_kv_bwd - e_k * v) / decay, (s_k_bwd - e_k) / decay
        num, den = s_kv_fwd + s_kv_bwd + e_u * v, s_k_fwd + s_k_bwd + e_u
        y = (num / (den + 1e-8)) * r
        tl.store(y_ptr + t * stride_y_t + n_offsets, y.to(Y.dtype.element_ty), mask=n_offsets < N)
        s_kv_fwd, s_k_fwd = s_kv_fwd * decay + e_k * v, s_k_fwd * decay + e_k

@triton.jit
def bi_wkv_backward_kernel(
    R, K, V, W, U, GY, GR, GK, GV, GW, GU,
    stride_r_b, stride_r_h, stride_r_t, stride_r_n,
    stride_k_b, stride_k_h, stride_k_t, stride_k_n,
    stride_v_b, stride_v_h, stride_v_t, stride_v_n,
    stride_w_h, stride_w_n, stride_u_h, stride_u_n,
    stride_gy_b, stride_gy_h, stride_gy_t, stride_gy_n,
    T: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    b_idx, h_idx = tl.program_id(0), tl.program_id(1)
    r_ptr, k_ptr, v_ptr = R + b_idx*stride_r_b + h_idx*stride_r_h, K + b_idx*stride_k_b + h_idx*stride_k_h, V + b_idx*stride_v_b + h_idx*stride_v_h
    w_ptr, u_ptr, gy_ptr = W + h_idx*stride_w_h, U + h_idx*stride_u_h, GY + b_idx*stride_gy_b + h_idx*stride_gy_h
    gr_ptr, gk_ptr, gv_ptr = GR + b_idx*stride_r_b + h_idx*stride_r_h, GK + b_idx*stride_k_b + h_idx*stride_k_h, GV + b_idx*stride_v_b + h_idx*stride_v_h
    gw_ptr, gu_ptr = GW + h_idx*stride_w_h, GU + h_idx*stride_u_h
    n_offsets = tl.arange(0, BLOCK_SIZE)
    
    s_kv_fwd, s_k_fwd, s_kv_bwd, s_k_bwd = (tl.zeros((T, BLOCK_SIZE), dtype=tl.float32) for _ in range(4))
    _s_kv_fwd, _s_k_fwd, _s_kv_bwd, _s_k_bwd = (tl.zeros([BLOCK_SIZE], dtype=tl.float32) for _ in range(4))
    
    w = tl.load(w_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
    decay = tl.exp(-tl.exp(w))

    for t in range(T):
        k, v = tl.load(k_ptr + t*stride_k_t + n_offsets, mask=n_offsets < N).to(tl.float32), tl.load(v_ptr + t*stride_v_t + n_offsets, mask=n_offsets < N).to(tl.float32)
        e_k = tl.exp(k)
        tl.store(s_kv_fwd + t*BLOCK_SIZE, _s_kv_fwd); tl.store(s_k_fwd + t*BLOCK_SIZE, _s_k_fwd)
        _s_kv_fwd, _s_k_fwd = _s_kv_fwd*decay + e_k*v, _s_k_fwd*decay + e_k

    for t in range(T - 1, -1, -1):
        k, v = tl.load(k_ptr + t*stride_k_t + n_offsets, mask=n_offsets < N).to(tl.float32), tl.load(v_ptr + t*stride_v_t + n_offsets, mask=n_offsets < N).to(tl.float32)
        e_k = tl.exp(k)
        tl.store(s_kv_bwd + t*BLOCK_SIZE, _s_kv_bwd); tl.store(s_k_bwd + t*BLOCK_SIZE, _s_k_bwd)
        _s_kv_bwd, _s_k_bwd = _s_kv_bwd*decay + e_k*v, _s_k_bwd*decay + e_k

    g_s_kv_fwd, g_s_k_fwd, g_s_kv_bwd, g_s_k_bwd = (tl.zeros([BLOCK_SIZE], dtype=tl.float32) for _ in range(4))
    gw, gu = tl.zeros([BLOCK_SIZE], dtype=tl.float32), tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for t in range(T - 1, -1, -1):
        r, k, v, u, gy = (tl.load(ptr + t*stride_t + n_offsets, mask=n_offsets < N).to(tl.float32) for ptr, stride_t in [(r_ptr, stride_r_t), (k_ptr, stride_k_t), (v_ptr, stride_v_t), (u_ptr, stride_u_t), (gy_ptr, stride_gy_t)])
        _s_kv_fwd, _s_k_fwd, _s_kv_bwd, _s_k_bwd = (tl.load(s + t*BLOCK_SIZE) for s in [s_kv_fwd, s_k_fwd, s_kv_bwd, s_k_bwd])
        e_k, e_u = tl.exp(k), tl.exp(u)
        num, den = _s_kv_fwd + _s_kv_bwd + e_u*v, _s_k_fwd + _s_k_bwd + e_u
        den_inv = 1.0 / (den + 1e-8)
        
        g_num, g_den = gy*r*den_inv, -gy*r*num*den_inv*den_inv
        gr = gy * (num * den_inv)
        tl.store(gr_ptr + t*stride_r_t + n_offsets, gr.to(GR.dtype.element_ty), mask=n_offsets < N)
        
        g_s_kv_fwd_t, g_s_k_fwd_t, g_s_kv_bwd_t, g_s_k_bwd_t = g_num, g_den, g_num, g_den
        gu += (g_num*v + g_den)*e_u
        gv_t = g_num*e_u + g_s_kv_fwd*e_k + g_s_kv_bwd*e_k
        tl.store(gv_ptr + t*stride_v_t + n_offsets, gv_t.to(GV.dtype.element_ty), mask=n_offsets < N)
        gk_t = (g_num*v + g_den)*e_u*e_k + g_s_kv_fwd*v*e_k + g_s_k_fwd*e_k + g_s_kv_bwd*v*e_k + g_s_k_bwd*e_k
        tl.store(gk_ptr + t*stride_k_t + n_offsets, gk_t.to(GK.dtype.element_ty), mask=n_offsets < N)
        
        g_s_kv_fwd, g_s_k_fwd, g_s_kv_bwd, g_s_k_bwd = (g*decay + g_t for g, g_t in [(g_s_kv_fwd, g_s_kv_fwd_t), (g_s_k_fwd, g_s_k_fwd_t), (g_s_kv_bwd, g_s_kv_bwd_t), (g_s_k_bwd, g_s_k_bwd_t)])
        gw += (g_s_kv_fwd*_s_kv_fwd + g_s_k_fwd*_s_k_fwd + g_s_kv_bwd*_s_kv_bwd + g_s_k_bwd*_s_k_bwd) * -decay * tl.exp(w)

    tl.store(gw_ptr + n_offsets, gw.to(GW.dtype.element_ty), mask=n_offsets < N)
    tl.store(gu_ptr + n_offsets, gu.to(GU.dtype.element_ty), mask=n_offsets < N)

class BiWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        B, T, C = r.shape; H = C // config.head_size; N = config.head_size
        r, k, v = (x.view(B, T, H, N).transpose(1, 2).contiguous() for x in (r, k, v))
        y = torch.empty_like(r)
        
        grid = (B, H)
        bi_wkv_forward_kernel[grid](
            r, k, v, w, u, y,
            r.stride(0), r.stride(1), r.stride(2), r.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), w.stride(0), w.stride(1), u.stride(0), u.stride(1),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            T=T, N=N, BLOCK_SIZE=N
        )
        ctx.save_for_backward(r, k, v, w, u)
        return y.transpose(1, 2).contiguous().view(B, T, C)

    @staticmethod
    def backward(ctx, gy):
        r, k, v, w, u = ctx.saved_tensors
        B, H, T, N = r.shape
        gy = gy.view(B, T, H, N).transpose(1, 2).contiguous()
        gr, gk, gv, gw, gu = torch.empty_like(r), torch.empty_like(k), torch.empty_like(v), torch.empty_like(w), torch.empty_like(u)
        
        grid = (B, H)
        bi_wkv_backward_kernel[grid](
            r, k, v, w, u, gy, gr, gk, gv, gw, gu,
            r.stride(0), r.stride(1), r.stride(2), r.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), w.stride(0), w.stride(1), u.stride(0), u.stride(1),
            gy.stride(0), gy.stride(1), gy.stride(2), gy.stride(3),
            T=T, N=N, BLOCK_SIZE=N
        )
        gr, gk, gv = (x.transpose(1, 2).contiguous().view(B, T, H*N) for x in (gr, gk, gv))
        return gr, gk, gv, gw, gu

# --- 3. Model Definition ---

class BiRWKVBlock(nn.Module):
    """A single block of the Bidirectional RWKV model."""
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        C = config.n_embd
        
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        
        self.att_receptance = nn.Linear(C, C, bias=False)
        self.att_key = nn.Linear(C, C, bias=False)
        self.att_value = nn.Linear(C, C, bias=False)
        self.att_output = nn.Linear(C, C, bias=False)
        self.att_decay = nn.Parameter(torch.zeros(C))
        self.att_u = nn.Parameter(torch.zeros(C))
        
        self.ffn_key = nn.Linear(C, C * 4, bias=False)
        self.ffn_value = nn.Linear(C * 4, C, bias=False)
        self.ffn_receptance = nn.Linear(C, C, bias=False)

class BiRWKVModel(nn.Module):
    """The full Bidirectional RWKV model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([BiRWKVBlock(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        x = self.word_embeddings(idx)

        for block in self.blocks:
            # Time-mixing (Attention)
            ln_x = block.ln1(x)
            r = torch.sigmoid(block.att_receptance(ln_x))
            k = block.att_key(ln_x)
            v = block.att_value(ln_x)
            
            # Reshape decay and u for the kernel
            H = self.config.n_embd // self.config.head_size
            decay = block.att_decay.view(H, self.config.head_size)
            u = block.att_u.view(H, self.config.head_size)
            
            x_att = BiWKV.apply(r, k, v, decay, u)
            x = x + block.att_output(x_att)
            
            # Channel-mixing (FFN)
            ln_x = block.ln2(x)
            r_ffn = torch.sigmoid(block.ffn_receptance(ln_x))
            k_ffn = torch.relu(block.ffn_key(ln_x)) ** 2
            x = x + r_ffn * block.ffn_value(k_ffn)
            
        x = self.ln_out(x)
        logits = self.head(x)
        return logits

# --- 4. Main Training Script ---

def main():
    # Initialize Accelerator
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=config.gradient_accumulation_steps)
    accelerator = Accelerator(mixed_precision=config.mixed_precision, deepspeed_plugin=deepspeed_plugin)
    
    set_seed(config.seed)

    # Load Tokenizer
    # For this example, we assume a RoBERTa-style tokenizer is available.
    # You should train your own BPE tokenizer for a real project.
    if not os.path.exists(config.tokenizer_path):
        print(f"Downloading tokenizer...")
        from transformers import RobertaTokenizer
        tokenizer_hf = RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer_hf.save_pretrained(".", legacy_format=False)
        os.rename("tokenizer.json", config.tokenizer_path)
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    # Load and process dataset
    accelerator.print("Loading and processing dataset...")
    raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name)

    def tokenize_function(examples):
        return tokenizer.encode_batch(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw_datasets["train"].column_names,
    )
    
    # Group texts into blocks of sequence_length
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // config.sequence_length) * config.sequence_length
        result = {
            k: [t[i : i + config.sequence_length] for i in range(0, total_length, config.sequence_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )
    
    train_dataset = lm_datasets["train"]
    
    # Data Collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.per_device_train_batch_size
    )

    # Initialize Model, Optimizer, and LR Scheduler
    accelerator.print("Initializing model...")
    model = BiRWKVModel(config)
    accelerator.print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Start Training
    accelerator.print("***** Starting Pre-training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {config.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {config.per_device_train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {config.per_device_train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps}")
    accelerator.print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(config.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(batch["input_ids"])
                loss = F.cross_entropy(outputs.view(-1, config.vocab_size), batch["labels"].view(-1))
                
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if accelerator.is_local_main_process:
                    # Log metrics (e.g., to wandb or console)
                    # For simplicity, we just print loss here.
                    if completed_steps % 100 == 0:
                        print(f"Step {completed_steps}: Loss {loss.item():.4f}")

                if completed_steps % 5000 == 0:
                    # Save checkpoint
                    accelerator.print(f"Saving checkpoint at step {completed_steps}")
                    save_path = os.path.join(config.output_dir, f"checkpoint-{completed_steps}")
                    accelerator.save_state(save_path)

    accelerator.print("***** Training finished *****")
    # Save final model
    if accelerator.is_local_main_process:
        final_path = os.path.join(config.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))


if __name__ == "__main__":
    main()