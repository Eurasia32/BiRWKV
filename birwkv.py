from __future__ import annotations

import logging
import math
import os
from abc import abstractmethod
from typing import (
    NamedTuple,
    Optional,
    Tuple,
    Union,
    List,
    Dict,
    Any,
)
from dataclasses import dataclass, field
from functools import partial
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
try:
    biwkv_cuda = load(
        name="bi_wkv",
        sources=[
            os.path.join(module_path, "cuda/bi_wkv.cpp"),
            os.path.join(module_path, "cuda/bi_wkv_kernel.cu"),
        ],
        verbose = True,
        extra_cuda_cflags=['-res-usage', '--maxrregcount 60', 
        '--use_fast_math', '-O3', '-Xptxas -O3', 
        '-gencode arch=compute_80,code=sm_80']
    )
    BIWKV_CUDA_AVAILABLE = True
    logging.info("Successfully loaded CUDA Kernel")
except Exception as e:
    BIWKV_CUDA_AVAILABLE = False
    logging.warning(f"Could not load BiRWKV-LLADA CUDA kernel. Falling back to slow PyTorch implementation. Error: {e}")

class BiWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = biwkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)

def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())

def BiShift(input, shift_token=1, split_ratio=1/2):
    """
    对输入的1D序列张量进行双向平移 (Bi-directional Shift)。

    Args:
        input (torch.Tensor): 输入张量，形状为 (B, N, C)
                                     B: 批量大小 (Batch size)
                                     N: 序列长度 (Sequence length)
                                     C: 特征/通道维度 (Embedding dimension)
        shift_token (int): 在序列维度上平移的步长（token数量）。
        split_ratio (float): 两方向平移的通道所占的比例。例如，0.5表示一半通道参与平移。
    
    Returns:
        torch.Tensor: 平移后的张量，形状与输入相同。
    """
    assert 0 < split_ratio <= 1
    B, N, C = input.shape
    output = torch.zeros_like(input)
    output[:, shift_token:N, 0:int(C*split_ratio)] = input[:, 0:N-shift_token, 0:int(C*split_ratio)]
    output[:, 0:N-shift_token, int(C*split_ratio):int(C*split_ratio*2)] = input[:, shift_token:N, int(C*split_ratio):int(C*split_ratio*2)]
    output[:, :, int(C*split_ratio*2):] = input[:, :, int(C*split_ratio*2):]
    return output

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, split_ratio=1/2, 
                shift_token=1, k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights()
        self.shift_token = shift_token
        if shift_token > 0:
            self.shift_func = BiShift()
            self.split_ratio = split_ratio
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.value.scale_init = 1

    def _init_weights(self):
        with torch.no_grad(): # fancy init
            ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
            # fancy time_decay
            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
            # fancy time_mix
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, split_ratio=1/2, 
                shift_token=1, hidden_rate=4, k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights()
        self.shift_token = shift_token
        if shift_token > 0:
            self.shift_func = BiShift()
            self.split_ratio = split_ratio
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

        self.key.scale_init = 1

    def _init_weights(self):
        with torch.no_grad():
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

    def forward(self, x, patch_resolution=None):
        if self.shift_token > 0:
            xx = self.shift_func(x, self.shift_token, self.split_ratio)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

class BiRWKVBlock(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, split_ratio=1/2, shift_token=1,
                hidden_rate=4, init_values=None,
                post_norm=False, k_norm=True, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        ## Dropout?

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id,
                                    split_ratio, shift_token,
                                    k_norm=k_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id,
                                    split_ratio, shift_token, hidden_rate,
                                    k_norm=k_norm)

        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp
    def forward(self, x):
        