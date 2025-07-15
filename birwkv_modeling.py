#
# modeling_birwkv_llada.py
#
# This script implements the "BiRWKV-LLADA" model, a diffusion language model
# that combines bidirectional RWKV attention with LLADA's diffusion framework.
# Features: BiWKV attention, BiShift mechanism, diffusion noise scheduling,
# and time-conditioned generation with custom CUDA kernels.
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import numpy as np

# --- Load Custom CUDA Kernel ---
# This will compile the C++/CUDA code on the fly for BiRWKV-LLADA diffusion operations.
# Assumes the .cpp and .cu files are in the same directory as this script.
module_path = os.path.dirname(__file__)
try:
    birwkv_llada_cuda = load(
        name="birwkv_llada",
        sources=[
            os.path.join(module_path, "cuda/rwkv8/birwkv_op.cpp"),
            os.path.join(module_path, "cuda/rwkv8/birwkv_kernel.cu"),
        ],
        verbose=False, # Set to True for compilation details
    )
    BIWKV_LLADA_CUDA_AVAILABLE = True
    logging.info("Successfully loaded BiRWKV-LLADA CUDA kernel.")
except Exception as e:
    BIWKV_LLADA_CUDA_AVAILABLE = False
    logging.warning(f"Could not load BiRWKV-LLADA CUDA kernel. Falling back to slow PyTorch implementation. Error: {e}")


# --- Diffusion and Model Configuration ---
@dataclass
class DiffusionConfig:
    """Configuration for diffusion process in BiRWKV-LLADA."""
    num_diffusion_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "cosine", "sigmoid"
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    
@dataclass
class ModelConfig:
    """Configuration for BiRWKV-LLADA model."""
    # Model architecture
    d_model: int = 1024
    n_layers: int = 24
    vocab_size: int = 50257
    max_sequence_length: int = 2048
    
    # BiRWKV specific
    mlp_ratio: float = 2.5
    mlp_hidden_size: Optional[int] = None
    
    # Diffusion
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    
    # Layer norm and initialization
    layer_norm_type: str = "rms"
    layer_norm_with_affine: bool = True
    bias_for_layer_norm: Optional[bool] = None
    include_bias: bool = False
    rms_norm_eps: float = 1e-5
    
    # Initialization
    init_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    init_fn: str = "normal"
    init_std: float = 0.02
    init_cutoff_factor: Optional[float] = None
    
    # Regularization
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0
    residual_dropout: float = 0.0
    
    # Other
    weight_tying: bool = True
    scale_logits: bool = False
    input_emb_norm: bool = False
    embedding_size: Optional[int] = None
    
class InitFnType(StrEnum):
    normal = "normal"
    mitchell = "mitchell"
    kaiming_normal = "kaiming_normal"
    fan_in = "fan_in"
    full_megatron = "full_megatron"
    
class LayerNormType(StrEnum):
    default = "default"
    low_precision = "low_precision"
    rms = "rms"
    gemma_rms = "gemma_rms"
    
class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"
    
class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    
class BufferCache:
    """Simple buffer cache for attention computations."""
    def __init__(self):
        self._cache = {}
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._cache.get(key)
        
    def __setitem__(self, key: str, value: torch.Tensor):
        self._cache[key] = value
        
    def __getitem__(self, key: str) -> torch.Tensor:
        return self._cache[key]
# --- Diffusion Noise Scheduling ---
class DiffusionScheduler:
    """Noise scheduler for LLADA diffusion process."""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_train_timesteps = config.num_diffusion_steps
        
        # Create beta schedule
        if config.beta_schedule == "linear":
            self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps)
        elif config.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(config.num_diffusion_steps)
        elif config.beta_schedule == "sigmoid":
            self.betas = self._sigmoid_beta_schedule(config.num_diffusion_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")
            
        # Compute derived values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For denoising
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
        
    def _sigmoid_beta_schedule(self, timesteps: int, start: float = -3, end: float = 3) -> torch.Tensor:
        betas = torch.linspace(start, end, timesteps)
        return torch.sigmoid(betas) * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
        
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples according to the noise schedule."""
        device = original_samples.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
        
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity for v-prediction parameterization."""
        device = sample.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


# --- Time Embedding for Diffusion ---
class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion process."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half)
        freqs = freqs.to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
        
class TimeConditionedLinear(nn.Module):
    """Linear layer conditioned on timestep embedding."""
    
    def __init__(self, in_features: int, out_features: int, time_dim: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_dim = time_dim
        
        # Main linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Time conditioning
        self.time_proj = nn.Linear(time_dim, out_features, bias=False)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Apply main linear transformation
        out = self.linear(x)
        
        # Add time conditioning
        time_cond = self.time_proj(time_emb)
        if len(time_cond.shape) == 2:  # [batch_size, features]
            time_cond = time_cond.unsqueeze(1)  # [batch_size, 1, features]
            
        out = out + time_cond
        return out


log = logging.getLogger(__name__)

# --- Enhanced BiShift with Time Conditioning ---
class BiShiftLLADA(nn.Module):
    """Bidirectional shift mechanism enhanced for LLADA diffusion."""
    
    def __init__(self, config: ModelConfig, time_dim: int = 128):
        super().__init__()
        self.config = config
        self.time_dim = time_dim
        
        # Base shift parameters
        self.mu = nn.Parameter(torch.zeros(1, 1, config.d_model))
        
        # Time-conditioned shift parameters
        self.time_shift_proj = nn.Linear(time_dim, config.d_model, bias=False)
        self.time_scale_proj = nn.Linear(time_dim, config.d_model, bias=False)
        
        # Learnable shift patterns for different time steps
        self.shift_patterns = nn.Parameter(torch.randn(3, config.d_model) * 0.02)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Time-conditioned shift parameters
        time_shift = self.time_shift_proj(time_emb).unsqueeze(1)  # [B, 1, C]
        time_scale = torch.sigmoid(self.time_scale_proj(time_emb)).unsqueeze(1)  # [B, 1, C]
        
        # Create padded tensor for shifting
        x_padded = F.pad(x, (0, 0, 1, 1), mode='replicate')
        
        # Multi-direction shifts
        x_left = x_padded[:, 2:, :]  # Left shift
        x_right = x_padded[:, :-2, :]  # Right shift
        x_center = x_padded[:, 1:-1, :]  # Center (original)
        
        # Dynamic channel splitting based on time
        C_third = C // 3
        C_remainder = C % 3
        
        # Split channels for different shift directions
        if C_remainder == 0:
            splits = [C_third, C_third, C_third]
        elif C_remainder == 1:
            splits = [C_third + 1, C_third, C_third]
        else:
            splits = [C_third + 1, C_third + 1, C_third]
            
        # Apply different shifts to different channel groups
        x_shifted_parts = []
        start_idx = 0
        
        for i, split_size in enumerate(splits):
            end_idx = start_idx + split_size
            
            if i == 0:
                shift_part = x_left[:, :, start_idx:end_idx]
            elif i == 1:
                shift_part = x_right[:, :, start_idx:end_idx]
            else:
                shift_part = x_center[:, :, start_idx:end_idx]
                
            # Apply time-conditioned pattern
            pattern = self.shift_patterns[i, start_idx:end_idx].unsqueeze(0).unsqueeze(0).to(x.device)
            shift_part = shift_part + time_shift[:, :, start_idx:end_idx] * pattern
            
            x_shifted_parts.append(shift_part)
            start_idx = end_idx
            
        x_shifted = torch.cat(x_shifted_parts, dim=2)
        
        # Time-conditioned mixing
        base_mu = torch.sigmoid(self.mu)
        time_modulated_mu = base_mu * time_scale
        
        return x + time_modulated_mu * (x_shifted - x)


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))

def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)

def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )

class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore

class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)

class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x

class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x

class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")

class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias

# --- Output Classes for LLADA ---
class BiRWKVLLADAOutput(NamedTuple):
    """Output from BiRWKV-LLADA model."""
    logits: torch.FloatTensor
    """Denoised logits of shape (batch_size, seq_len, vocab_size)."""
    
    noise_pred: Optional[torch.FloatTensor]
    """Predicted noise of shape (batch_size, seq_len, d_model)."""
    
    v_pred: Optional[torch.FloatTensor]
    """Velocity prediction for v-parameterization."""
    
    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    """Hidden states from each layer."""
    
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """Attention keys and values from each block."""
    
class BiRWKVLLADAGenerateOutput(NamedTuple):
    """Output from BiRWKV-LLADA generation."""
    token_ids: torch.LongTensor
    """Generated token IDs of shape (batch_size, beam_size, max_steps)."""
    
    scores: torch.FloatTensor
    """Generation scores of shape (batch_size, beam_size)."""
    
    intermediate_states: Optional[List[torch.Tensor]]
    """Intermediate denoising states for analysis."""

# --- Main BiRWKV-LLADA Model ---
class BiRWKVLLADAModel(nn.Module):
    """BiRWKV-LLADA: A diffusion language model with bidirectional RWKV attention."""
    
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        
        # Time embedding for diffusion
        self.time_dim = 128
        self.time_embedding = TimestepEmbedding(self.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        
        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(config.diffusion)
        
        # Validate embedding size
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )
        
        # Activation checkpointing
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)
        
        # Enable optimized attention backends
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        
        # Core transformer components
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNormBase.build(config),
            )
        )
        
        # BiRWKV-LLADA blocks
        blocks = [BiRWKVLLADABlock(i, config, self.time_dim) for i in range(config.n_layers)]
        self.transformer.update({"blocks": nn.ModuleList(blocks)})
        
        # Positional embeddings
        self.transformer.update(
            {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
        )
        
        # Output projection
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        
        # Diffusion-specific heads
        self.noise_head = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_head = nn.Linear(config.d_model, config.d_model, bias=False) if config.diffusion.prediction_type == "v_prediction" else None
        
        # Noise prediction conditioning
        self.noise_conditioning = nn.Linear(self.time_dim, config.d_model, bias=False)
        
        # Initialize parameters
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
    
    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        """Set activation checkpointing strategy."""
        self.activation_checkpointing_strategy = strategy
        for block in self.transformer.blocks:
            block.set_activation_checkpointing(strategy)
    
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def reset_parameters(self):
        """Initialize model parameters."""
        log.info("Initializing BiRWKV-LLADA model parameters...")
        
        # Embeddings
        init_weights(
            self.config,
            self.transformer.wte,
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)
        
        # Layer norm
        self.transformer.ln_f.reset_parameters()
        
        # Output projection
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)
        
        # Time embedding MLP
        for module in self.time_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Diffusion heads
        nn.init.normal_(self.noise_head.weight, std=0.02)
        nn.init.normal_(self.noise_conditioning.weight, std=0.02)
        if self.v_head is not None:
            nn.init.normal_(self.v_head.weight, std=0.02)
        
        # Initialize blocks
        for block in self.transformer.blocks:
            block.reset_parameters()

    def forward(
        self,
        input_ids: torch.LongTensor,
        timesteps: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_noise_pred: bool = False,
        return_v_pred: bool = False,
    ) -> BiRWKVLLADAOutput:
        """
        Forward pass for BiRWKV-LLADA diffusion model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            timesteps: Diffusion timesteps of shape (batch_size,)
            input_embeddings: Pre-computed embeddings of shape (batch_size, seq_len, d_model)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            attention_bias: Attention bias tensor
            past_key_values: Cached key-value pairs for generation
            use_cache: Whether to return attention cache
            last_logits_only: Whether to compute only the last token logits
            output_hidden_states: Whether to return hidden states
            return_noise_pred: Whether to return noise predictions
            return_v_pred: Whether to return velocity predictions
        """
        
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        if past_key_values:
            assert len(past_key_values) == self.config.n_layers
        
        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        
        # Handle timesteps for diffusion
        if timesteps is None:
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        else:
            # Ensure timesteps are on the same device as input_ids
            timesteps = timesteps.to(input_ids.device)
        
        # Time embedding
        time_emb = self.time_embedding(timesteps)  # [B, time_dim]
        time_emb = self.time_mlp(time_emb)  # [B, time_dim]
        
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)
        
        # Get embeddings
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings
        
        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)
        
        # Positional embeddings
        pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x = pos_emb + x
        
        # Embedding dropout
        x = self.transformer.emb_drop(x)
        
        # Process attention mask
        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None
        
        # Prepare attention bias
        if (
            attention_bias is not None
            or attention_mask is not None
            or past_key_values is not None
        ):
            if attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)
            
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)
        
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        all_hidden_states = []
        
        # Apply BiRWKV-LLADA blocks
        for block_idx, block in enumerate(self.transformer.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)
            
            layer_past = None if past_key_values is None else past_key_values[block_idx]
            
            # Activation checkpointing
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                x, cache = self._activation_checkpoint_fn(
                    block, x, time_emb, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                )
            else:
                x, cache = block(x, time_emb, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        
        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)
        
        # Compute logits
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))
        
        # Diffusion-specific predictions
        noise_pred = None
        v_pred = None
        
        if return_noise_pred:
            # Add time conditioning to noise prediction
            noise_conditioning = self.noise_conditioning(time_emb).unsqueeze(1)
            conditioned_x = x + noise_conditioning
            noise_pred = self.noise_head(conditioned_x)
        
        if return_v_pred and self.v_head is not None:
            v_pred = self.v_head(x)
        
        return BiRWKVLLADAOutput(
            logits=logits,
            noise_pred=noise_pred,
            v_pred=v_pred,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attn_key_values=attn_key_values,
        )

    def compute_diffusion_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss for BiRWKV-LLADA.
        
        Args:
            input_ids: Clean input token IDs
            attention_mask: Attention mask
            noise: Noise to add (if None, will be sampled)
            timesteps: Diffusion timesteps (if None, will be sampled)
            
        Returns:
            Dictionary containing loss components
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Sample timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.num_train_timesteps, (batch_size,), device=device
            )
        else:
            # Ensure timesteps are on the correct device
            timesteps = timesteps.to(device)
        
        # Get clean embeddings
        clean_embeddings = self.transformer.wte(input_ids)
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(clean_embeddings)
        else:
            # Ensure noise is on the correct device
            noise = noise.to(device)
        
        # Add noise according to schedule
        noisy_embeddings = self.scheduler.add_noise(clean_embeddings, noise, timesteps)
        
        # Forward pass with noisy embeddings
        output = self.forward(
            input_ids=input_ids,
            timesteps=timesteps,
            input_embeddings=noisy_embeddings,
            attention_mask=attention_mask,
            return_noise_pred=True,
            return_v_pred=(self.config.diffusion.prediction_type == "v_prediction"),
        )
        
        losses = {}
        
        # Compute prediction loss based on parameterization
        if self.config.diffusion.prediction_type == "epsilon":
            target = noise
            prediction = output.noise_pred
        elif self.config.diffusion.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(clean_embeddings, noise, timesteps)
            prediction = output.v_pred
        elif self.config.diffusion.prediction_type == "sample":
            target = clean_embeddings
            prediction = output.noise_pred
        else:
            raise ValueError(f"Unknown prediction type: {self.config.diffusion.prediction_type}")
        
        # MSE loss on predictions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            mse_loss = F.mse_loss(prediction * mask, target * mask, reduction="none")
            mse_loss = mse_loss.sum() / mask.sum()
        else:
            mse_loss = F.mse_loss(prediction, target)
        
        losses["diffusion_loss"] = mse_loss
        
        # Optional: Add token prediction loss for better language modeling
        if hasattr(self.config, "token_loss_weight") and self.config.token_loss_weight > 0:
            token_loss = F.cross_entropy(
                output.logits.view(-1, output.logits.size(-1)),
                input_ids.view(-1),
                ignore_index=-100
            )
            losses["token_loss"] = token_loss
            losses["total_loss"] = mse_loss + self.config.token_loss_weight * token_loss
        else:
            losses["total_loss"] = mse_loss
        
        return losses
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples using DDPM sampling.
        
        Args:
            shape: Shape of samples to generate (batch_size, seq_len)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            generator: Random number generator
            prompt_embeds: Optional prompt embeddings for conditioning
            attention_mask: Attention mask
            
        Returns:
            Generated token embeddings
        """
        batch_size, seq_len = shape
        device = self.device
        
        # Sample pure noise
        sample = torch.randn(
            (batch_size, seq_len, self.config.d_model),
            device=device,
            generator=generator,
            dtype=self.transformer.wte.weight.dtype
        )
        
        # Create sampling timesteps
        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
        )
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            timestep_batch = t.repeat(batch_size)
            
            # Predict noise
            if prompt_embeds is not None:
                # Use prompt embeddings as input
                model_input = sample
                dummy_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            else:
                # Convert embeddings to approximate token IDs for processing
                dummy_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
                model_input = sample
            
            output = self.forward(
                input_ids=dummy_ids,
                timesteps=timestep_batch,
                input_embeddings=model_input,
                attention_mask=attention_mask,
                return_noise_pred=True,
            )
            
            noise_pred = output.noise_pred
            
            # Compute previous sample using DDPM formula
            if i < len(timesteps) - 1:
                # Not the last step
                prev_timestep = timesteps[i + 1]
                alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device)
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep].to(device)
                
                beta_prod_t = 1 - alpha_prod_t
                
                # Compute predicted original sample
                pred_original_sample = (sample - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
                
                # Compute previous sample
                pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * noise_pred
                sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
            else:
                # Last step: return predicted clean sample
                alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device)
                beta_prod_t = 1 - alpha_prod_t
                sample = (sample - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        
        return sample
    
    def generate_tokens(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        num_inference_steps: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """
        Generate tokens using diffusion-based sampling.
        
        Args:
            prompt_ids: Input prompt token IDs
            max_new_tokens: Maximum number of new tokens to generate
            num_inference_steps: Number of diffusion steps
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token IDs
        """
        batch_size, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        
        # Generate embeddings for new tokens
        new_embeddings = self.sample(
            shape=(batch_size, max_new_tokens),
            num_inference_steps=num_inference_steps,
        )
        
        # Convert embeddings to logits
        if self.config.weight_tying:
            logits = F.linear(new_embeddings, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(new_embeddings)
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample tokens
        probs = F.softmax(logits, dim=-1)
        new_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, max_new_tokens)
        
        # Concatenate with prompt
        generated_ids = torch.cat([prompt_ids, new_tokens], dim=1)
        
# --- Layer Normalization Implementations ---
class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> "LayerNormBase":
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

class LayerNorm(LayerNormBase):
    """Default LayerNorm implementation."""

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)

class RMSLayerNorm(LayerNormBase):
    """RMS layer norm implementation."""

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x

class GemmaRMSLayerNorm(LayerNormBase):
    """Gemma RMS layer norm implementation."""

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x

# --- Missing Helper Components ---
class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"

class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )

def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """Initialize weights of a linear or embedding module."""
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))

def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """Modify tensor in place to replace infinities with finite values."""
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)

def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)

def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias

# --- Enhanced BiRWKV Function with Diffusion Support ---
class BiRWKVLLADAFunction(torch.autograd.Function):
    """Enhanced BiRWKV function with diffusion and time conditioning support."""
    
    @staticmethod
    def forward(ctx, r, k, v, w, u, time_emb):
        B, T, C = r.shape
        r, k, v, w, u = r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), u.contiguous()
        time_emb = time_emb.contiguous()
        
        # Extended workspace for diffusion operations
        workspace = torch.empty(B * T * C * 6, device=r.device, dtype=torch.float32)
        y = torch.empty((B, T, C), device=r.device, dtype=r.dtype)
        
        if BIWKV_LLADA_CUDA_AVAILABLE:
            birwkv_llada_cuda.forward(r, k, v, w, u, time_emb, y, workspace)
        else:
            # Fallback PyTorch implementation
            y = BiRWKVLLADAFunction._pytorch_forward(r, k, v, w, u, time_emb)
            
        ctx.save_for_backward(r, k, v, w, u, time_emb, y, workspace)
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        r, k, v, w, u, time_emb, y, workspace = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        
        grad_r = torch.empty_like(r)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_w = torch.zeros_like(w, dtype=torch.float32)
        grad_u = torch.zeros_like(u, dtype=torch.float32)
        grad_time_emb = torch.zeros_like(time_emb, dtype=torch.float32)
        
        if BIWKV_LLADA_CUDA_AVAILABLE:
            birwkv_llada_cuda.backward(
                r, k, v, w, u, time_emb, y, grad_y, workspace,
                grad_r, grad_k, grad_v, grad_w, grad_u, grad_time_emb
            )
        else:
            # Fallback PyTorch backward
            grad_r, grad_k, grad_v, grad_w, grad_u, grad_time_emb = BiRWKVLLADAFunction._pytorch_backward(
                r, k, v, w, u, time_emb, grad_y
            )
            
        return grad_r, grad_k, grad_v, grad_w, grad_u, grad_time_emb
    
    @staticmethod
    def _pytorch_forward(r, k, v, w, u, time_emb):
        """PyTorch fallback implementation with basic time conditioning."""
        B, T, C = r.shape
        
        # Time conditioning factor
        time_factor = torch.sigmoid(time_emb.sum(dim=-1, keepdim=True).unsqueeze(-1))  # [B, 1, 1]
        
        # Enhanced decay with time conditioning
        decay = torch.exp(-torch.exp(w.float())) * (0.5 + 0.5 * time_factor)
        
        # Forward pass
        a_fwd = torch.zeros(B, C, device=r.device, dtype=torch.float32)
        b_fwd = torch.zeros(B, C, device=r.device, dtype=torch.float32)
        num_fwd = torch.zeros_like(k, dtype=torch.float32)
        den_fwd = torch.zeros_like(k, dtype=torch.float32)
        
        for t in range(T):
            kt = k[:, t, :].float()
            vt = v[:, t, :].float()
            exp_kt = torch.exp(torch.clamp(kt, max=30.0))
            
            a_fwd = a_fwd * decay.squeeze() + exp_kt * vt
            b_fwd = b_fwd * decay.squeeze() + exp_kt
            num_fwd[:, t, :] = a_fwd
            den_fwd[:, t, :] = b_fwd
            
        # Backward pass
        a_bwd = torch.zeros(B, C, device=r.device, dtype=torch.float32)
        b_bwd = torch.zeros(B, C, device=r.device, dtype=torch.float32)
        num_bwd = torch.zeros_like(k, dtype=torch.float32)
        den_bwd = torch.zeros_like(k, dtype=torch.float32)
        
        for t in reversed(range(T)):
            kt = k[:, t, :].float()
            vt = v[:, t, :].float()
            exp_kt = torch.exp(torch.clamp(kt, max=30.0))
            
            a_bwd = a_bwd * decay.squeeze() + exp_kt * vt
            b_bwd = b_bwd * decay.squeeze() + exp_kt
            num_bwd[:, t, :] = a_bwd
            den_bwd[:, t, :] = b_bwd
            
        # Combine forward and backward
        kt_f = k.float()
        vt_f = v.float()
        rt_f = r.float()
        
        exp_kt = torch.exp(torch.clamp(kt_f, max=30.0))
        exp_uk = torch.exp(torch.clamp(u.float() + kt_f, max=30.0))
        
        num = num_fwd + num_bwd - (exp_kt * vt_f) + (exp_uk * vt_f)
        den = den_fwd + den_bwd - exp_kt + exp_uk
        
        # Time-conditioned output
        base_output = torch.sigmoid(rt_f) * (num / torch.clamp(den, min=1e-8))
        time_modulated_output = base_output * (0.8 + 0.2 * time_factor)
        
        return time_modulated_output.to(r.dtype)
    
    @staticmethod
    def _pytorch_backward(r, k, v, w, u, time_emb, grad_y):
        """Simplified PyTorch backward implementation."""
        # This is a simplified version - full implementation would require
        # proper gradient computation through the bidirectional RNN
        grad_r = torch.zeros_like(r)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_w = torch.zeros_like(w, dtype=torch.float32)
        grad_u = torch.zeros_like(u, dtype=torch.float32)
        grad_time_emb = torch.zeros_like(time_emb, dtype=torch.float32)
        
        return grad_r, grad_k, grad_v, grad_w, grad_u, grad_time_emb

# --- Enhanced BiRWKV Attention with Time Conditioning ---
class BiRWKVLLADAAttention(nn.Module):
    """BiRWKV attention mechanism enhanced for LLADA diffusion."""
    
    def __init__(self, config: ModelConfig, layer_id: int, time_dim: int = 128):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.d_model = config.d_model
        self.time_dim = time_dim
        
        # Base parameters
        self.w = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.u = nn.Parameter(torch.zeros(1, 1, self.d_model))
        
        # Time conditioning for parameters
        self.time_w_proj = nn.Linear(time_dim, self.d_model, bias=False)
        self.time_u_proj = nn.Linear(time_dim, self.d_model, bias=False)
        
        # Layer-specific time conditioning
        self.layer_time_scale = nn.Parameter(torch.ones(1) * (1.0 / (layer_id + 1)))
        
    def forward(self, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, T, C = r.shape
        
        # Time-conditioned parameters
        time_w_delta = self.time_w_proj(time_emb).unsqueeze(1) * self.layer_time_scale
        time_u_delta = self.time_u_proj(time_emb).unsqueeze(1) * self.layer_time_scale
        
        w_conditioned = self.w + time_w_delta
        u_conditioned = self.u + time_u_delta
        
        if BIWKV_LLADA_CUDA_AVAILABLE and r.is_cuda:
            return BiRWKVLLADAFunction.apply(r, k, v, w_conditioned, u_conditioned, time_emb)
        else:
            # Fallback to PyTorch implementation
            return BiRWKVLLADAFunction._pytorch_forward(r, k, v, w_conditioned, u_conditioned, time_emb)

# --- Enhanced BiRWKV Block with LLADA Diffusion ---
class BiRWKVLLADABlock(nn.Module):
    """BiRWKV block enhanced for LLADA diffusion with time conditioning."""
    
    def __init__(self, layer_id: int, config: ModelConfig, time_dim: int = 128):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.time_dim = time_dim
        
        # Layer normalization
        self.ln1 = LayerNormBase.build(config)
        self.ln2 = LayerNormBase.build(config)
        
        # Enhanced BiShift with time conditioning
        self.shift1 = BiShiftLLADA(config, time_dim)
        self.shift2 = BiShiftLLADA(config, time_dim)
        
        # Attention projections with time conditioning
        self.r_proj = TimeConditionedLinear(config.d_model, config.d_model, time_dim, bias=False)
        self.k_proj = TimeConditionedLinear(config.d_model, config.d_model, time_dim, bias=False)
        self.v_proj = TimeConditionedLinear(config.d_model, config.d_model, time_dim, bias=False)
        
        # Enhanced attention mechanism
        self.attention = BiRWKVLLADAAttention(config, layer_id, time_dim)
        self.attn_out = TimeConditionedLinear(config.d_model, config.d_model, time_dim, bias=False)
        
        # MLP with time conditioning
        hidden_size = config.mlp_hidden_size if hasattr(config, 'mlp_hidden_size') and config.mlp_hidden_size is not None else int(config.mlp_ratio * config.d_model)
        self.ff_r_proj = TimeConditionedLinear(config.d_model, config.d_model, time_dim, bias=False)
        self.ff_k_proj = TimeConditionedLinear(config.d_model, hidden_size, time_dim, bias=False)
        self.ff_v_proj = TimeConditionedLinear(hidden_size, config.d_model, time_dim, bias=False)
        
        # Diffusion-specific components
        self.noise_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.time_gate = nn.Linear(time_dim, config.d_model, bias=False)
        
    def forward(
        self, 
        x: torch.Tensor,
        time_emb: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Time-conditioned gating
        time_gate = torch.sigmoid(self.time_gate(time_emb)).unsqueeze(1)  # [B, 1, C]
        
        # Attention block with time conditioning
        residual = x
        x_norm = self.ln1(x)
        x_shifted = self.shift1(x_norm, time_emb)
        
        # Time-conditioned projections
        r = self.r_proj(x_shifted, time_emb)
        k = self.k_proj(x_shifted, time_emb)
        v = self.v_proj(x_shifted, time_emb)
        
        # Enhanced attention with time conditioning
        attn_output = self.attention(r, k, v, time_emb)
        attn_output = self.attn_out(attn_output, time_emb)
        
        # Time-gated residual connection
        x = residual + time_gate * attn_output
        
        # MLP block with time conditioning
        residual = x
        x_norm = self.ln2(x)
        x_shifted = self.shift2(x_norm, time_emb)
        
        # Time-conditioned MLP
        r_ff = self.ff_r_proj(x_shifted, time_emb)
        k_ff = self.ff_k_proj(x_shifted, time_emb)
        
        # Enhanced gating mechanism
        gated_k = F.relu(k_ff) ** 2
        ffn_output = self.ff_v_proj(torch.sigmoid(r_ff) * gated_k, time_emb)
        
        # Time-gated residual connection
        x = residual + time_gate * ffn_output
        
        # Optional noise injection for training stability
        if self.training:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise
            
        return x, None
    
    def reset_parameters(self):
        """Initialize parameters for the block."""
        # Initialize layer norms
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        
        # Initialize time conditioning parameters
        nn.init.normal_(self.time_gate.weight, std=0.02)
        nn.init.constant_(self.noise_scale, 0.1)

# --- Module Export ---
__all__ = [
    "ModelConfig",
    "DiffusionConfig", 
    "DiffusionScheduler",
    "TimestepEmbedding",
    "TimeConditionedLinear",
    "BiShiftLLADA",
    "BiRWKVLLADAFunction",
    "BiRWKVLLADAAttention", 
    "BiRWKVLLADABlock",
    "BiRWKVLLADAModel",
    "BiRWKVLLADAOutput",
    "BiRWKVLLADAGenerateOutput",
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm", 
    "GemmaRMSLayerNorm",
]
