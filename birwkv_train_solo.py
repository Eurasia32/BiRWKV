#
# train_birwkv_llada.py
#
# Enhanced training script for BiRWKV-LLADA diffusion language model.
# This version supports diffusion training with time conditioning, noise scheduling,
# and multiple prediction parameterizations. Supports resuming training from checkpoints.
#
# 运行要求:
# 1. 安装 PyTorch, transformers, datasets, sentencepiece, ninja
#    pip install torch transformers datasets sentencepiece ninja
# 2. 将 birwkv_op.cpp 和 birwkv_kernel.cu 文件放置在同一目录下。
# 3. 确保您的环境已安装NVIDIA CUDA Toolkit和C++编译器(如g++)。
#
# 运行命令示例 (扩散训练从零开始):
# python birwkv_train.py --batch_size 8 --learning_rate 3e-4 --max_steps 50000 --diffusion_steps 1000
#
# 运行命令示例 (从检查点恢复):
# python birwkv_train.py --resume_from_checkpoint ./birwkv-llada-model/checkpoint-10000
#
# 运行命令示例 (v-prediction参数化):
# python birwkv_train.py --prediction_type v_prediction --beta_schedule cosine --token_loss_weight 0.1
#n

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm
import numpy as np

# Import our enhanced BiRWKV-LLADA model
try:
    from birwkv_modeling import (
        ModelConfig,
        DiffusionConfig,
        BiRWKVLLADAModel,
        BIWKV_LLADA_CUDA_AVAILABLE
    )
except ImportError:
    print("错误: 无法从 'birwkv_modeling.py' 导入模型。请确保该文件在同一目录下。")
    exit(1)

# --- 日志设置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# --- Enhanced Data Collator for Diffusion Training ---
class BiRWKVLLADADataCollator:
    """Data collator for BiRWKV-LLADA diffusion training."""
    
    def __init__(self, tokenizer, max_len: int, diffusion_mode: str = "mixed"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.diffusion_mode = diffusion_mode  # "diffusion", "token", "mixed"
        
        # Add special tokens if not present
        if '[MASK]' not in tokenizer.get_vocab():
            tokenizer.add_tokens(['[MASK]'])
        self.mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
        
    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Tokenize texts
        batch = self.tokenizer(
            [e["text"] for e in examples], 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len
        )
        
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        
        # Create labels for token prediction (if needed)
        labels = input_ids.clone()
        
        # For mixed training, sometimes do masking for token prediction
        if self.diffusion_mode in ["token", "mixed"]:
            if self.diffusion_mode == "mixed" and torch.rand(1).item() < 0.3:  # 30% token prediction
                # Traditional masked language modeling
                mask_ratio = 0.15 + torch.rand(1).item() * 0.35  # 15-50% masking
                non_padding_mask = (input_ids != self.tokenizer.pad_token_id)
                probability_matrix = torch.full(labels.shape, mask_ratio)
                masked_indices = torch.bernoulli(probability_matrix).bool() & non_padding_mask
                input_ids[masked_indices] = self.mask_token_id
                labels[~masked_indices] = -100
                
                batch["training_mode"] = "token"
            else:
                # Pure diffusion training - no masking
                labels = input_ids.clone()  # Keep original for diffusion
                batch["training_mode"] = "diffusion"
        else:
            # Pure diffusion mode
            batch["training_mode"] = "diffusion"
            
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        
        return batch


# --- Training Configuration ---
@dataclass
class TrainingConfig:
    """Training configuration for BiRWKV-LLADA."""
    # Model architecture
    d_model: int = 1024
    n_layers: int = 24
    mlp_ratio: float = 2.5
    
    # Diffusion parameters
    num_diffusion_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "cosine", "sigmoid"
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"
    
    # Training parameters
    seq_len: int = 512
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100000
    warmup_steps: int = 1000
    
    # Mixed training
    diffusion_mode: str = "mixed"  # "diffusion", "token", "mixed"
    token_loss_weight: float = 0.1  # Weight for token prediction loss in mixed mode
    
    # Other
    save_interval: int = 5000
    output_dir: str = "./birwkv-llada-model"
    resume_from_checkpoint: Optional[str] = None
    
    def to_model_config(self, vocab_size: int) -> ModelConfig:
        """Convert to ModelConfig."""
        diffusion_config = DiffusionConfig(
            num_diffusion_steps=self.num_diffusion_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
        )
        
        return ModelConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            vocab_size=vocab_size,
            max_sequence_length=self.seq_len,
            mlp_ratio=self.mlp_ratio,
            diffusion=diffusion_config,
        )


# --- Training Loop ---
def train_step(model, batch, config, device):
    """Single training step with diffusion loss."""
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    training_mode = batch.get("training_mode", ["diffusion"] * len(batch["input_ids"]))
    
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch.get("attention_mask", None)
    
    total_loss = 0.0
    loss_dict = {}
    
    # Check if this batch should use diffusion training
    if isinstance(training_mode, list):
        use_diffusion = training_mode[0] == "diffusion"
    else:
        use_diffusion = training_mode == "diffusion"
    
    if use_diffusion:
        # Diffusion training
        losses = model.compute_diffusion_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        diffusion_loss = losses["diffusion_loss"]
        total_loss += diffusion_loss
        loss_dict["diffusion_loss"] = diffusion_loss.item()
        
        # Optional token prediction loss
        if "token_loss" in losses and config.token_loss_weight > 0:
            token_loss = losses["token_loss"]
            total_loss += config.token_loss_weight * token_loss
            loss_dict["token_loss"] = token_loss.item()
    
    else:
        # Traditional token prediction training
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        token_loss = F.cross_entropy(
            output.logits.view(-1, output.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        total_loss = token_loss
        loss_dict["token_loss"] = token_loss.item()
    
    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict


# --- Main Training Function ---
def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")
    
    if BIWKV_LLADA_CUDA_AVAILABLE:
        log.info("BiRWKV-LLADA CUDA 核函数已加载")
    else:
        log.warning("BiRWKV-LLADA CUDA 核函数未加载，使用PyTorch实现")
    
    # Training configuration
    config = TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        num_diffusion_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        diffusion_mode=args.diffusion_mode,
        token_loss_weight=args.token_loss_weight,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if '[MASK]' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['[MASK]'])
    
    # Model setup
    model_config = config.to_model_config(vocab_size=len(tokenizer))
    model = BiRWKVLLADAModel(model_config)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"总参数量: {total_params / 1e6:.2f}M, 可训练参数: {trainable_params / 1e6:.2f}M")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay, 
        betas=(0.9, 0.95)
    )
    
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=config.warmup_steps, 
        num_training_steps=config.max_steps
    )
    
    completed_steps = 0
    
    # Load checkpoint if specified
    if config.resume_from_checkpoint:
        checkpoint_path = os.path.join(config.resume_from_checkpoint, "checkpoint.pt")
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
    
    # Dataset setup
    log.info("加载和预处理数据集...")
    raw_datasets = load_dataset("wikitext", "wikitext-103-v1", streaming=True)
    train_dataset = raw_datasets["train"]
    
    data_collator = BiRWKVLLADADataCollator(
        tokenizer, 
        max_len=config.seq_len,
        diffusion_mode=config.diffusion_mode
    )
    
    train_dataloader = DataLoader(
        train_dataset.with_format("torch"), 
        collate_fn=data_collator, 
        batch_size=config.batch_size
    )
    train_iterator = iter(train_dataloader)
    
    # Fast-forward data loader if resuming
    if completed_steps > 0:
        log.info(f"快进数据加载器 {completed_steps} 步...")
        for _ in tqdm(range(completed_steps), desc="快进数据"):
            try:
                next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                next(train_iterator)
    
    # Training loop
    model.train()
    progress_bar = tqdm(range(completed_steps, config.max_steps), desc="BiRWKV-LLADA 训练进度")
    
    # Running averages for logging
    running_losses = {}
    log_interval = 100
    
    while completed_steps < config.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        
        # Mixed precision training
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
            loss, loss_dict = train_step(model, batch, config, device)
        
        # Check for NaN
        if torch.isnan(loss):
            log.error("损失值为NaN，跳过此步骤。")
            continue
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Update running averages
        for key, value in loss_dict.items():
            if key not in running_losses:
                running_losses[key] = value
            else:
                running_losses[key] = 0.9 * running_losses[key] + 0.1 * value
        
        # Update progress
        progress_bar.update(1)
        
        # Log current losses
        log_dict = {
            "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
            "total": f"{running_losses.get('total_loss', 0):.4f}"
        }
        if "diffusion_loss" in running_losses:
            log_dict["diff"] = f"{running_losses['diffusion_loss']:.4f}"
        if "token_loss" in running_losses:
            log_dict["token"] = f"{running_losses['token_loss']:.4f}"
            
        progress_bar.set_postfix(log_dict)
        completed_steps += 1
        
        # Detailed logging
        if completed_steps % log_interval == 0:
            log_msg = f"Step {completed_steps}: "
            for key, value in running_losses.items():
                log_msg += f"{key}={value:.4f} "
            log_msg += f"lr={lr_scheduler.get_last_lr()[0]:.6f}"
            log.info(log_msg)
        
        # Save checkpoint
        if completed_steps % config.save_interval == 0:
            save_path = os.path.join(config.output_dir, f"checkpoint-{completed_steps}")
            os.makedirs(save_path, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'completed_steps': completed_steps,
                'config': config,
                'running_losses': running_losses,
            }
            
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))
            log.info(f"检查点已保存至 {save_path}")
    
    log.info("BiRWKV-LLADA 训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练BiRWKV-LLADA扩散语言模型")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=1024, help="模型隐藏维度")
    parser.add_argument("--n_layers", type=int, default=24, help="模型层数")
    parser.add_argument("--mlp_ratio", type=float, default=2.5, help="MLP隐藏层比例")
    
    # Diffusion parameters
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="扩散训练步数")
    parser.add_argument("--beta_schedule", type=str, default="linear", 
                       choices=["linear", "cosine", "sigmoid"], help="噪声调度类型")
    parser.add_argument("--prediction_type", type=str, default="epsilon",
                       choices=["epsilon", "v_prediction", "sample"], help="预测参数化类型")
    
    # Training parameters
    parser.add_argument("--seq_len", type=int, default=512, help="输入序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--max_steps", type=int, default=100000, help="总训练步数")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="学习率预热步数")
    
    # Mixed training
    parser.add_argument("--diffusion_mode", type=str, default="mixed",
                       choices=["diffusion", "token", "mixed"], help="训练模式")
    parser.add_argument("--token_loss_weight", type=float, default=0.1, help="令牌预测损失权重")
    
    # Other
    parser.add_argument("--save_interval", type=int, default=5000, help="模型保存间隔")
    parser.add_argument("--output_dir", type=str, default="./birwkv-llada-model", help="模型输出目录")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="检查点恢复路径")
    
    args = parser.parse_args()
    main(args)