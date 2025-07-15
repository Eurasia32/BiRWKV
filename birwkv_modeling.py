#
# modeling_birwkv.py
#
# This script implements the "BiRWKV" model, a fusion of the LLaDA diffusion
# framework and the efficient, bidirectional attention mechanism from Vision-RWKV.
# It is designed to be a drop-in replacement for the original LLaDA model architecture
# and now calls a custom high-performance CUDA kernel for its attention mechanism.
#

from __future__ import annotations

import logging
import math
import os
from abc import abstractmethod
from typing import (
    NamedTuple,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# --- Load Custom CUDA Kernel ---
# This will compile the C++/CUDA code on the fly.
# Assumes the .cpp and .cu files are in the same directory as this script.
module_path = os.path.dirname(__file__)
try:
    birwkv_cuda = load(
        name="birwkv",
        sources=[
            os.path.join(module_path, "cuda/rwkv8/birwkv_op.cpp"),
            os.path.join(module_path, "cuda/rwkv8/birwkv_kernel.cu"),
        ],
        verbose=False, # Set to True for compilation details
    )
    BIWKV_CUDA_AVAILABLE = True
    logging.info("Successfully loaded BiRWKV CUDA kernel.")
except Exception as e:
    BIWKV_CUDA_AVAILABLE = False
    logging.warning(f"Could not load BiRWKV CUDA kernel. Falling back to slow PyTorch implementation. Error: {e}")


# --- Redefined Enums and Base Classes for self-containment ---
class StrEnum(str):
    def __new__(cls, value, *args, **kwargs):
        return super().__new__(cls, value)

class BlockType(StrEnum):
    birwkv = "birwkv"

log = logging.getLogger(__name__)

class BufferCache(dict):
    pass

class LayerNormBase(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.eps = kwargs.get('eps', 1e-5)
        self.normalized_shape = (kwargs.get('size') or config.d_model,)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    @classmethod
    def build(cls, config, **kwargs):
        return RMSLayerNorm(config, **kwargs)

class RMSLayerNorm(LayerNormBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x + self.bias

class LLaDABlock(nn.Module):
    def __init__(self, layer_id: int, config, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.__cache = cache

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config, cache: BufferCache):
        if config.block_type == BlockType.birwkv:
            return BiRWKVBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Block type '{config.block_type}' not supported.")

# --------------------------------------------------------------------------- #
#                   BiRWKV IMPLEMENTATION STARTS HERE                         #
# --------------------------------------------------------------------------- #

class BiShift(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mu = nn.Parameter(torch.zeros(1, 1, config.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, (0, 0, 1, 1), mode='replicate')
        x_left = x_padded[:, 2:, :]
        x_right = x_padded[:, :-2, :]
        C_half = self.config.d_model // 2
        x_shifted = torch.cat([x_left[:, :, :C_half], x_right[:, :, C_half:]], dim=2)
        return x + torch.sigmoid(self.mu) * (x_shifted - x)

class BiRWKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        B, T, C = r.shape
        r, k, v, w, u = r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), u.contiguous()
        
        # Workspace for storing intermediate RNN states (4 of them)
        workspace = torch.empty(B * T * C * 4, device=r.device, dtype=torch.float32)
        y = torch.empty((B, T, C), device=r.device, dtype=r.dtype)
        
        birwkv_cuda.forward(r, k, v, w, u, y, workspace)
        
        ctx.save_for_backward(r, k, v, w, u, y, workspace)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        r, k, v, w, u, y, workspace = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        
        grad_r = torch.empty_like(r)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_w = torch.zeros_like(w, dtype=torch.float32) # Gradients for w and u are accumulated
        grad_u = torch.zeros_like(u, dtype=torch.float32)
        
        birwkv_cuda.backward(
            r, k, v, w, u, y, grad_y, workspace,
            grad_r, grad_k, grad_v, grad_w, grad_u
        )
        return grad_r, grad_k, grad_v, grad_w, grad_u

class BiRWKVAttention(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.d_model = config.d_model
        
        self.w = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.u = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def forward(self, r, k, v):
        if BIWKV_CUDA_AVAILABLE and r.is_cuda:
            return BiRWKVFunction.apply(r, k, v, self.w, self.u)
        else:
            # Fallback to slow PyTorch implementation
            raise RuntimeError("BiRWKV CUDA kernel not available. Cannot run.")

class BiRWKVBlock(LLaDABlock):
    def __init__(self, layer_id: int, config, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        
        self.ln1 = LayerNormBase.build(config)
        self.shift1 = BiShift(config)
        self.r_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attention = BiRWKVAttention(config, layer_id)
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.ln2 = LayerNormBase.build(config)
        self.shift2 = BiShift(config)
        hidden_size = config.mlp_hidden_size if hasattr(config, 'mlp_hidden_size') and config.mlp_hidden_size is not None else int(config.mlp_ratio * config.d_model)
        self.ff_r_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ff_k_proj = nn.Linear(config.d_model, hidden_size, bias=False)
        self.ff_v_proj = nn.Linear(hidden_size, config.d_model, bias=False)

    def forward(
        self, 
        x: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        residual = x
        x_norm = self.ln1(x)
        x_shifted = self.shift1(x_norm)
        r, k, v = self.r_proj(x_shifted), self.k_proj(x_shifted), self.v_proj(x_shifted)
        attn_output = self.attention(r, k, v)
        x = residual + self.attn_out(attn_output)

        residual = x
        x_norm = self.ln2(x)
        x_shifted = self.shift2(x_norm)
        r_ff = self.ff_r_proj(x_shifted)
        k_ff = self.ff_k_proj(x_shifted)
        gated_k = F.relu(k_ff) ** 2
        ffn_output = self.ff_v_proj(torch.sigmoid(r_ff) * gated_k)
        x = residual + ffn_output

        return x, None
