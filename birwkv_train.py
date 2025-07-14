#
# train_birwkv.py
#
# A complete script to pre-train a BiRWKV-LLaDA model from scratch.
# This version supports resuming training from a saved checkpoint.
#
# 运行要求:
# 1. 安装 PyTorch, transformers, datasets, sentencepiece, ninja
#    pip install torch transformers datasets sentencepiece ninja
# 2. 将 birwkv_op.cpp 和 birwkv_kernel.cu 文件放置在同一目录下。
# 3. 确保您的环境已安装NVIDIA CUDA Toolkit和C++编译器(如g++)。
#
# 运行命令示例 (从零开始):
# python train_birwkv.py --batch_size 8 --learning_rate 3e-4 --max_steps 50000
#
# 运行命令示例 (从检查点恢复):
# python train_birwkv.py --resume_from_checkpoint ./birwkv-model/checkpoint-10000
#

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm

# --- 日志设置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# --- 加载自定义CUDA核函数 ---
module_path = os.path.dirname(__file__)
try:
    birwkv_cuda = load(
        name="birwkv",
        sources=[
            os.path.join(module_path, "birwkv_op.cpp"),
            os.path.join(module_path, "birwkv_kernel.cu"),
        ],
        verbose=True,
    )
    BIWKV_CUDA_AVAILABLE = True
    log.info("成功加载 BiRWKV CUDA 核函数。")
except Exception as e:
    BIWKV_CUDA_AVAILABLE = False
    log.warning(f"无法加载 BiRWKV CUDA 核函数，将回退到缓慢的PyTorch实现。错误: {e}")


# --- 模型配置 ---
@dataclass
class BiRWKVConfig:
    """Configuration for the BiRWKV model."""
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 32
    mlp_ratio: float = 2.5
    block_type: str = "birwkv"
    
    def count_parameters(self):
        hidden_size = int(self.d_model * self.mlp_ratio)
        params_per_block = (4 * self.d_model**2) + (2 * self.d_model * hidden_size + hidden_size * self.d_model)
        total_params = self.n_layers * params_per_block
        total_params += self.vocab_size * self.d_model + 2 * self.d_model
        return total_params


# --- 模型定义 ---

class RMSLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(x.dtype)

class BiShift(nn.Module):
    def __init__(self, config: BiRWKVConfig):
        super().__init__()
        self.config = config
        self.mu = nn.Parameter(torch.zeros(1, 1, config.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, (0, 0, 1, 1), mode='replicate')
        x_left, x_right = x_padded[:, 2:, :], x_padded[:, :-2, :]
        C_half = self.config.d_model // 2
        x_shifted = torch.cat([x_left[:, :, :C_half], x_right[:, :, C_half:]], dim=2)
        return x + torch.sigmoid(self.mu) * (x_shifted - x)

class BiRWKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        B, T, C = r.shape
        r, k, v, w, u = r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), u.contiguous()
        workspace = torch.empty(B * T * C * 4, device=r.device, dtype=torch.float32)
        y = torch.empty((B, T, C), device=r.device, dtype=r.dtype)
        birwkv_cuda.forward(r, k, v, w, u, y, workspace)
        ctx.save_for_backward(r, k, v, w, u, y, workspace)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        r, k, v, w, u, y, workspace = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        grad_r, grad_k, grad_v = torch.empty_like(r), torch.empty_like(k), torch.empty_like(v)
        grad_w, grad_u = torch.zeros_like(w, dtype=torch.float32), torch.zeros_like(u, dtype=torch.float32)
        birwkv_cuda.backward(r, k, v, w, u, y, grad_y, workspace, grad_r, grad_k, grad_v, grad_w, grad_u)
        return grad_r, grad_k, grad_v, grad_w, grad_u

class BiRWKVAttention(nn.Module):
    def __init__(self, config: BiRWKVConfig):
        super().__init__()
        self.config = config
        self.w = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.u = nn.Parameter(torch.zeros(1, 1, config.d_model))

    def forward(self, r, k, v):
        if BIWKV_CUDA_AVAILABLE and r.is_cuda:
            return BiRWKVFunction.apply(r, k, v, self.w, self.u)
        else:
            return self.pytorch_forward(r, k, v)
    
    def pytorch_forward(self, r, k, v):
        B, T, C = r.shape
        decay = torch.exp(-torch.exp(self.w.float()))
        a_fwd, b_fwd = torch.zeros(B, C, device=r.device, dtype=torch.float32), torch.zeros(B, C, device=r.device, dtype=torch.float32)
        num_fwd, den_fwd = torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(k, dtype=torch.float32)
        for t in range(T):
            kt, vt, exp_kt = k[:, t, :].float(), v[:, t, :].float(), torch.exp(k[:, t, :].float())
            a_fwd, b_fwd = a_fwd * decay + exp_kt * vt, b_fwd * decay + exp_kt
            num_fwd[:, t, :], den_fwd[:, t, :] = a_fwd, b_fwd
        a_bwd, b_bwd = torch.zeros(B, C, device=r.device, dtype=torch.float32), torch.zeros(B, C, device=r.device, dtype=torch.float32)
        num_bwd, den_bwd = torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(k, dtype=torch.float32)
        for t in reversed(range(T)):
            kt, vt, exp_kt = k[:, t, :].float(), v[:, t, :].float(), torch.exp(k[:, t, :].float())
            a_bwd, b_bwd = a_bwd * decay + exp_kt * vt, b_bwd * decay + exp_kt
            num_bwd[:, t, :], den_bwd[:, t, :] = a_bwd, b_bwd
        kt_f, vt_f, rt_f = k.float(), v.float(), r.float()
        exp_kt, exp_uk = torch.exp(kt_f), torch.exp(self.u.float() + kt_f)
        num, den = num_fwd + num_bwd - (exp_kt * vt_f) + (exp_uk * vt_f), den_fwd + den_bwd - exp_kt + exp_uk
        return (torch.sigmoid(rt_f) * (num / torch.clamp(den, min=1e-8))).to(r.dtype)

class BiRWKVBlock(nn.Module):
    def __init__(self, config: BiRWKVConfig):
        super().__init__()
        self.ln1 = RMSLayerNorm(config.d_model)
        self.shift1 = BiShift(config)
        self.r_proj, self.k_proj, self.v_proj = (nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(3))
        self.attention = BiRWKVAttention(config)
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ln2 = RMSLayerNorm(config.d_model)
        self.shift2 = BiShift(config)
        hidden_size = int(config.d_model * config.mlp_ratio)
        self.ff_r_proj = nn.Linear(config.d_model, hidden_size, bias=False)
        self.ff_k_proj = nn.Linear(config.d_model, hidden_size, bias=False)
        self.ff_v_proj = nn.Linear(hidden_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = self.shift1(x)
        r, k, v = self.r_proj(x), self.k_proj(x), self.v_proj(x)
        x = self.attention(r, k, v)
        x = residual + self.attn_out(x)
        residual = x
        x = self.ln2(x)
        x = self.shift2(x)
        r_ff, k_ff = self.ff_r_proj(x), self.ff_k_proj(x)
        gated_k = F.relu(k_ff) ** 2
        x = self.ff_v_proj(torch.sigmoid(r_ff) * gated_k)
        x = residual + x
        return x

class BiRWKVModel(nn.Module):
    def __init__(self, config: BiRWKVConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([BiRWKVBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSLayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# --- 数据处理 ---
class LLaDADataCollator:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
        self.max_len = max_len

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.tokenizer([e["text"] for e in examples], return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_len)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        mask_ratio = torch.rand(1).item()
        non_padding_mask = (input_ids != self.tokenizer.pad_token_id)
        probability_matrix = torch.full(labels.shape, mask_ratio)
        masked_indices = torch.bernoulli(probability_matrix).bool() & non_padding_mask
        input_ids[masked_indices] = self.mask_token_id
        labels[~masked_indices] = -100
        batch["input_ids"], batch["labels"] = input_ids, labels
        return batch

# --- 主训练函数 ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if '[MASK]' not in tokenizer.get_vocab(): tokenizer.add_tokens(['[MASK]'])
    
    config = BiRWKVConfig(vocab_size=len(tokenizer))
    model = BiRWKVModel(config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    completed_steps = 0
    
    # --- 新增: 加载检查点逻辑 ---
    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.resume_from_checkpoint, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            log.info(f"从检查点恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            completed_steps = checkpoint['completed_steps']
            log.info(f"已恢复至步骤 {completed_steps}")
        else:
            log.warning(f"检查点未找到: {checkpoint_path}。将从零开始训练。")

    log.info(f"模型已创建，总参数量: {config.count_parameters() / 1e6:.2f}M")

    log.info("加载和预处理数据集...")
    raw_datasets = load_dataset("wikitext", "wikitext-103-v1", streaming=True)
    train_dataset = raw_datasets["train"]
    data_collator = LLaDADataCollator(tokenizer, max_len=args.seq_len)
    train_dataloader = DataLoader(train_dataset.with_format("torch"), collate_fn=data_collator, batch_size=args.batch_size)
    train_iterator = iter(train_dataloader)

    # --- 新增: 快进数据加载器 ---
    if completed_steps > 0:
        log.info(f"快进数据加载器 {completed_steps} 步...")
        for _ in tqdm(range(completed_steps), desc="快进数据"):
            next(train_iterator)

    model.train()
    progress_bar = tqdm(range(completed_steps, args.max_steps), desc="训练进度")
    
    while completed_steps < args.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch["input_ids"])
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), batch["labels"].view(-1))
        
        #if torch.isnan(loss):
            #log.error("损失值为NaN，停止训练。")
            #break
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}"})
        completed_steps += 1
        
        if completed_steps % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
            os.makedirs(save_path, exist_ok=True)
            # --- 新增: 保存完整的检查点 ---
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'completed_steps': completed_steps,
                'args': args
            }
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))
            log.info(f"完整检查点已保存至 {save_path}")

    log.info("训练完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从零开始或从检查点恢复训练BiRWKV-LLaDA模型")
    parser.add_argument("--seq_len", type=int, default=512, help="输入序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="最大学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--max_steps", type=int, default=100000, help="总训练步数")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="学习率预热步数")
    parser.add_argument("--save_interval", type=int, default=5000, help="模型保存间隔")
    parser.add_argument("--output_dir", type=str, default="./birwkv-model", help="模型输出目录")
    # --- 新增: 恢复训练的参数 ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="要从中恢复训练的检查点目录路径。")
    
    args = parser.parse_args()
    main(args)