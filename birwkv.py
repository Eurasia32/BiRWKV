########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#
# This version is adapted for PARALLEL, NON-AUTOREGRESSIVE GENERATION.
# It uses the bidirectional architecture to function like a Masked Language Model
# or a Diffusion-like model, generating text via iterative refinement.
#
# Key Changes:
# 1. New `generate_parallel` function implements an iterative refinement loop.
# 2. Generation starts from a sequence of [MASK] tokens and is refined over several steps.
# 3. A confidence-based masking schedule (cosine annealing) is used to decide which tokens to
#    re-predict at each step, mimicking diffusion models.
# 4. The model is no longer an encoder but a parallel generator. The main example demonstrates this new capability.
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, os, math
from typing import List
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F

# Attempt to import Triton
try:
    import triton
    import triton.language as tl
    print("Triton imported successfully.")
    HAS_TRITON = True
except ImportError:
    print("Triton not found. Please install it with 'pip install triton'")
    HAS_TRITON = False

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

########################################################################################################
#
print('\nBidirectional RWKV-8 for Parallel Generation - https://github.com/BlinkDL/RWKV-LM')
print('\nNOTE: this is very inefficient (loads all weights to VRAM), you can actually prefetch .enn from RAM/SSD\n')
#
########################################################################################################

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/rwkv-8-pile
args.MODEL_NAME = "/mnt/g/_rwkv_checkpt/rwkv-8-pile-rc00-20250508"

# WORLD_MODE = True
WORLD_MODE = False # pile models

args.n_layer = 12
args.n_embd = 768
args.vocab_size = 65536 if WORLD_MODE else 50304
args.head_size = 64

# --- Parameters for Parallel Generation ---
prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was that the unicorns spoke perfect English."
GENERATION_LENGTH = 256 # Total length of the sequence to generate
GENERATION_STEPS = 12   # Number of refinement iterations
TEMPERATURE = 1.0       # Sampling temperature
TOP_P = 0.9             # Nucleus sampling p
MASK_TOKEN_ID = 0       # ID for the [MASK] token (0 is common, e.g., in BERT)

########################################################################################################

DTYPE = torch.half
HEAD_SIZE = args.head_size

if HAS_TRITON:
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
        """
        Triton kernel for the forward pass of Bidirectional WKV.
        """
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)
        
        # Pointers to the head's data
        r_ptr = R + b_idx * stride_r_b + h_idx * stride_r_h
        k_ptr = K + b_idx * stride_k_b + h_idx * stride_k_h
        v_ptr = V + b_idx * stride_v_b + h_idx * stride_v_h
        w_ptr = W + h_idx * stride_w_h
        u_ptr = U + h_idx * stride_u_h
        y_ptr = Y + b_idx * stride_y_b + h_idx * stride_y_h

        # Head-wise state variables
        s_kv_fwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        s_k_fwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        s_kv_bwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        s_k_bwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        n_offsets = tl.arange(0, BLOCK_SIZE)

        # Pre-calculate backward states (second pass from T-1 to 0)
        for t in range(T - 1, -1, -1):
            k = tl.load(k_ptr + t * stride_k_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + t * stride_v_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
            
            e_k = tl.exp(k)
            decay = tl.exp(-tl.exp(w))
            
            s_kv_bwd = s_kv_bwd * decay + e_k * v
            s_k_bwd = s_k_bwd * decay + e_k

        # Main forward pass (from 0 to T-1)
        for t in range(0, T):
            k = tl.load(k_ptr + t * stride_k_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + t * stride_v_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
            r = tl.load(r_ptr + t * stride_r_t + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)

            e_k = tl.exp(k)
            e_u = tl.exp(u)
            decay = tl.exp(-tl.exp(w))

            # Update backward state by removing current token
            s_kv_bwd = (s_kv_bwd - e_k * v) / decay
            s_k_bwd = (s_k_bwd - e_k) / decay
            
            # Calculate numerator and denominator
            num = s_kv_fwd + s_kv_bwd + e_u * v
            den = s_k_fwd + s_k_bwd + e_u
            
            # Compute output and apply receptance
            y = (num / (den + 1e-8)) * r
            tl.store(y_ptr + t * stride_y_t + n_offsets, y.to(Y.dtype.element_ty), mask=n_offsets < N)

            # Update forward state for next token
            s_kv_fwd = s_kv_fwd * decay + e_k * v
            s_k_fwd = s_k_fwd * decay + e_k

    @triton.jit
    def bi_wkv_backward_kernel(
        R, K, V, W, U, GY,
        GR, GK, GV, GW, GU,
        stride_r_b, stride_r_h, stride_r_t, stride_r_n,
        stride_k_b, stride_k_h, stride_k_t, stride_k_n,
        stride_v_b, stride_v_h, stride_v_t, stride_v_n,
        stride_w_h, stride_w_n,
        stride_u_h, stride_u_n,
        stride_gy_b, stride_gy_h, stride_gy_t, stride_gy_n,
        T: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
        """
        Triton kernel for the backward pass of Bidirectional WKV.
        This is a full implementation.
        """
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)
        
        # Pointers
        r_ptr = R + b_idx * stride_r_b + h_idx * stride_r_h
        k_ptr = K + b_idx * stride_k_b + h_idx * stride_k_h
        v_ptr = V + b_idx * stride_v_b + h_idx * stride_v_h
        w_ptr = W + h_idx * stride_w_h
        u_ptr = U + h_idx * stride_u_h
        gy_ptr = GY + b_idx * stride_gy_b + h_idx * stride_gy_h
        
        gr_ptr = GR + b_idx * stride_r_b + h_idx * stride_r_h
        gk_ptr = GK + b_idx * stride_k_b + h_idx * stride_k_h
        gv_ptr = GV + b_idx * stride_v_b + h_idx * stride_v_h
        gw_ptr = GW + h_idx * stride_w_h
        gu_ptr = GU + h_idx * stride_u_h

        n_offsets = tl.arange(0, BLOCK_SIZE)

        # --- Step 1: Recalculate forward and backward states from scratch ---
        # This is necessary because we can't store all intermediate states.
        s_kv_fwd = tl.zeros((T, BLOCK_SIZE), dtype=tl.float32)
        s_k_fwd = tl.zeros((T, BLOCK_SIZE), dtype=tl.float32)
        s_kv_bwd = tl.zeros((T, BLOCK_SIZE), dtype=tl.float32)
        s_k_bwd = tl.zeros((T, BLOCK_SIZE), dtype=tl.float32)
        
        # Temp states for calculation
        _s_kv_fwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        _s_k_fwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        _s_kv_bwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        _s_k_bwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        w = tl.load(w_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
        decay = tl.exp(-tl.exp(w))

        for t in range(T):
            k = tl.load(k_ptr + t * stride_k_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            v = tl.load(v_ptr + t * stride_v_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            e_k = tl.exp(k)
            tl.store(s_kv_fwd + t * BLOCK_SIZE, _s_kv_fwd)
            tl.store(s_k_fwd + t * BLOCK_SIZE, _s_k_fwd)
            _s_kv_fwd = _s_kv_fwd * decay + e_k * v
            _s_k_fwd = _s_k_fwd * decay + e_k

        for t in range(T - 1, -1, -1):
            k = tl.load(k_ptr + t * stride_k_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            v = tl.load(v_ptr + t * stride_v_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            e_k = tl.exp(k)
            tl.store(s_kv_bwd + t * BLOCK_SIZE, _s_kv_bwd)
            tl.store(s_k_bwd + t * BLOCK_SIZE, _s_k_bwd)
            _s_kv_bwd = _s_kv_bwd * decay + e_k * v
            _s_k_bwd = _s_k_bwd * decay + e_k

        # --- Step 2: Backward pass to compute gradients ---
        # Gradient states for the recurrence
        g_s_kv_fwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        g_s_k_fwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        g_s_kv_bwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        g_s_k_bwd = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Accumulators for gw and gu
        gw = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        gu = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for t in range(T - 1, -1, -1):
            r = tl.load(r_ptr + t * stride_r_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            k = tl.load(k_ptr + t * stride_k_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            v = tl.load(v_ptr + t * stride_v_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            u = tl.load(u_ptr + n_offsets, mask=n_offsets < N).to(tl.float32)
            gy = tl.load(gy_ptr + t * stride_gy_t + n_offsets, mask=n_offsets < N).to(tl.float32)
            
            _s_kv_fwd = tl.load(s_kv_fwd + t * BLOCK_SIZE)
            _s_k_fwd = tl.load(s_k_fwd + t * BLOCK_SIZE)
            _s_kv_bwd = tl.load(s_kv_bwd + t * BLOCK_SIZE)
            _s_k_bwd = tl.load(s_k_bwd + t * BLOCK_SIZE)

            e_k = tl.exp(k)
            e_u = tl.exp(u)
            
            num = _s_kv_fwd + _s_kv_bwd + e_u * v
            den = _s_k_fwd + _s_k_bwd + e_u
            den_inv = 1.0 / (den + 1e-8)
            y = (num * den_inv) * r

            # Gradients of the output y w.r.t. its components
            g_y = gy
            g_num = g_y * r * den_inv
            g_den = -g_y * r * num * den_inv * den_inv
            
            # Gradient for R
            gr = g_y * (num * den_inv)
            tl.store(gr_ptr + t * stride_r_t + n_offsets, gr.to(GR.dtype.element_ty), mask=n_offsets < N)

            # Accumulate gradients for recurrent states
            g_s_kv_fwd_t = g_num
            g_s_k_fwd_t = g_den
            g_s_kv_bwd_t = g_num
            g_s_k_bwd_t = g_den
            
            # Gradient for U
            g_u_t = (g_num * v + g_den) * e_u
            gu += g_u_t

            # Gradient for V
            g_v_t = g_num * e_u + g_s_kv_fwd * e_k + g_s_kv_bwd * e_k
            tl.store(gv_ptr + t * stride_v_t + n_offsets, g_v_t.to(GV.dtype.element_ty), mask=n_offsets < N)

            # Gradient for K
            g_k_t = (g_num * v + g_den) * e_u * e_k + g_s_kv_fwd * v * e_k + g_s_k_fwd * e_k + g_s_kv_bwd * v * e_k + g_s_k_bwd * e_k
            tl.store(gk_ptr + t * stride_k_t + n_offsets, g_k_t.to(GK.dtype.element_ty), mask=n_offsets < N)
            
            # Propagate gradients through time
            g_s_kv_fwd = g_s_kv_fwd * decay + g_s_kv_fwd_t
            g_s_k_fwd = g_s_k_fwd * decay + g_s_k_fwd_t
            g_s_kv_bwd = g_s_kv_bwd * decay + g_s_kv_bwd_t
            g_s_k_bwd = g_s_k_bwd * decay + g_s_k_bwd_t

            # Gradient for W
            g_w_t = (g_s_kv_fwd * _s_kv_fwd + g_s_k_fwd * _s_k_fwd + g_s_kv_bwd * _s_kv_bwd + g_s_k_bwd * _s_k_bwd) * -decay * tl.exp(w)
            gw += g_w_t

        tl.store(gw_ptr + n_offsets, gw.to(GW.dtype.element_ty), mask=n_offsets < N)
        tl.store(gu_ptr + n_offsets, gu.to(GU.dtype.element_ty), mask=n_offsets < N)

class BI_WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        B, T, C = r.shape
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        
        # Reshape for kernel: B, T, C -> B, H, T, N
        r = r.view(B, T, H, N).transpose(1, 2).contiguous()
        k = k.view(B, T, H, N).transpose(1, 2).contiguous()
        v = v.view(B, T, H, N).transpose(1, 2).contiguous()
        
        y = torch.empty_like(r)
        
        grid = (B, H)
        bi_wkv_forward_kernel[grid](
            r, k, v, w, u, y,
            r.stride(0), r.stride(1), r.stride(2), r.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            w.stride(0), w.stride(1),
            u.stride(0), u.stride(1),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            T=T, N=N, BLOCK_SIZE=N # BLOCK_SIZE must be a power of 2 <= 1024
        )
        
        ctx.save_for_backward(r, k, v, w, u)
        
        # Reshape back: B, H, T, N -> B, T, C
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y

    @staticmethod
    def backward(ctx, gy):
        r, k, v, w, u = ctx.saved_tensors
        B, H, T, N = r.shape
        
        # Reshape grad for kernel: B, T, C -> B, H, T, N
        gy = gy.view(B, T, H, N).transpose(1, 2).contiguous()
        
        gr = torch.empty_like(r)
        gk = torch.empty_like(k)
        gv = torch.empty_like(v)
        gw = torch.empty_like(w)
        gu = torch.empty_like(u)
        
        grid = (B, H)
        bi_wkv_backward_kernel[grid](
            r, k, v, w, u, gy,
            gr, gk, gv, gw, gu,
            r.stride(0), r.stride(1), r.stride(2), r.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            w.stride(0), w.stride(1),
            u.stride(0), u.stride(1),
            gy.stride(0), gy.stride(1), gy.stride(2), gy.stride(3),
            T=T, N=N, BLOCK_SIZE=N
        )
        
        # Reshape grads back: B, H, T, N -> B, T, C
        gr = gr.transpose(1, 2).contiguous().view(B, T, H*N)
        gk = gk.transpose(1, 2).contiguous().view(B, T, H*N)
        gv = gv.transpose(1, 2).contiguous().view(B, T, H*N)
        
        return gr, gk, gv, gw, gu

def BI_WKV_OP(r, k, v, w, u):
    return BI_WKV.apply(r, k, v, w, u)

########################################################################################################

class RWKV_BiDir_x080(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        
        # Load weights and convert to a parameter dictionary
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        
        # Convert to a nn.ParameterDict for proper module registration
        self.z = nn.ParameterDict()

        self.n_head, self.head_size = w['blocks.0.att.r_k'].shape

        keys = list(w.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                w[k] = w[k].t()
            
            param_name = k.replace('.', '_') # nn.ParameterDict doesn't like dots
            self.z[param_name] = nn.Parameter(w[k].squeeze().to(dtype=DTYPE))

            if k.endswith('att.r_k'):
                 self.z[param_name] = nn.Parameter(w[k].flatten().to(dtype=DTYPE))

        assert self.head_size == args.head_size
        
        # Add and initialize the new learnable parameter 'u' for each layer
        for i in range(args.n_layer):
            self.z[f'blocks_{i}_att_u'] = nn.Parameter(torch.zeros_like(self.z[f'blocks_{i}_att_r_k']))
            # Rename original decay 'w' to avoid conflict
            if f'blocks_{i}_att_w' in self.z:
                self.z[f'blocks_{i}_att_decay'] = self.z.pop(f'blocks_{i}_att_w')

        # Pre-process embedding layer
        self.z.blocks_0_ln0_weight = nn.Parameter(w['blocks.0.ln0.weight'])
        self.z.blocks_0_ln0_bias = nn.Parameter(w['blocks.0.ln0.bias'])
        self.z.emb_weight = nn.Parameter(F.layer_norm(w['emb.weight'], (args.n_embd,), weight=self.z.blocks_0_ln0_weight, bias=self.z.blocks_0_ln0_bias))

    def forward(self, idx):
        B, T = idx.shape
        C = self.n_embd
        
        x = self.z.emb_weight[idx]

        # State for CMix (FFN)
        ffn_x_prev = torch.zeros((B, C), dtype=DTYPE, device=x.device)

        for i in range(self.n_layer):
            # Bidirectional Time-Mix (Attention)
            ln1_w = self.z[f'blocks_{i}_ln1_weight']
            ln1_b = self.z[f'blocks_{i}_ln1_bias']
            xx = F.layer_norm(x, (C,), weight=ln1_w, bias=ln1_b)
            
            r = xx @ self.z[f'blocks_{i}_att_receptance_weight']
            k = xx @ self.z[f'blocks_{i}_att_key_weight']
            v = xx @ self.z[f'blocks_{i}_att_value_weight']
            
            xx = BI_WKV_OP(r, k, v, self.z[f'blocks_{i}_att_decay'], self.z[f'blocks_{i}_att_u'])
            xx = xx @ self.z[f'blocks_{i}_att_output_weight']
            x = x + xx

            # Channel-Mix (FFN)
            ln2_w = self.z[f'blocks_{i}_ln2_weight']
            ln2_b = self.z[f'blocks_{i}_ln2_bias']
            xx = F.layer_norm(x, (C,), weight=ln2_w, bias=ln2_b)
            
            # Get enn.weight for the specific indices
            enn_w = self.z[f'blocks_{i}_ffn_enn_weight'][idx]
            
            xx, ffn_x_prev = RWKV_x080_CMix_seq(xx, ffn_x_prev, 
                                                self.z[f'blocks_{i}_ffn_x_k'], 
                                                self.z[f'blocks_{i}_ffn_key_weight'], 
                                                self.z[f'blocks_{i}_ffn_value_weight'], 
                                                enn_w)
            x = x + xx
        
        x = F.layer_norm(x, (C,), weight=self.z.ln_out_weight, bias=self.z.ln_out_bias)
        x = x @ self.z.head_weight
        return x

# This is the original Channel-Mix module, it remains the same
@MyStatic
def RWKV_x080_CMix_seq(x, x_prev, x_k, K_, V_, E_):
    B, T, C = x.shape
    x_shifted = torch.cat((x_prev.unsqueeze(1), x[:, :-1, :]), dim=1)
    
    xx = x - x_shifted
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    
    output = (k @ V_) * E_
    return output, x[:, -1, :]

########################################################################################################
#
# The new parallel generation logic
#
########################################################################################################

def sample_logits(logits, temperature=1.0, top_p=0.0):
    """
    Samples from logits with temperature and nucleus sampling.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    probs = F.softmax(logits.float(), dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0
    
    # Renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return torch.multinomial(probs, 1).squeeze(-1)

def generate_parallel(model, tokenizer, prompt, length, steps, temp, top_p):
    """
    Generates text using iterative refinement (mask-predict style).
    """
    model.eval()
    
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) > length:
        raise ValueError("Prompt is longer than the desired generation length.")

    # 1. Initialize the sequence with prompt and MASK tokens
    masked_sequence = prompt_tokens + [MASK_TOKEN_ID] * (length - len(prompt_tokens))
    tokens = torch.tensor(masked_sequence, dtype=torch.long, device='cuda').unsqueeze(0)
    
    # A mask to indicate which tokens are part of the prompt and should not be changed
    prompt_mask = torch.ones_like(tokens, dtype=torch.bool)
    prompt_mask[:, :len(prompt_tokens)] = False

    print("--- Starting Parallel Generation ---")
    print(f"Initial sequence: {tokenizer.decode(tokens.squeeze().tolist())}\n")

    with torch.no_grad():
        for step in range(steps):
            # 2. Get logits for the entire sequence in parallel
            logits = model(tokens)
            
            # 3. Sample new tokens for all positions
            sampled_tokens = sample_logits(logits, temperature=temp, top_p=top_p)
            
            # 4. Calculate confidence scores for the sampled tokens
            probs = F.softmax(logits, dim=-1)
            sampled_probs = torch.gather(probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Set confidence of prompt tokens to infinity to ensure they are always kept
            sampled_probs.masked_fill_(~prompt_mask, float('inf'))

            # 5. Masking schedule: Cosine annealing
            # Start by keeping very few tokens, end by keeping almost all
            ratio = (step + 1) / steps
            keep_ratio = math.cos(ratio * math.pi / 2) # Starts at 1, ends at 0
            num_to_mask = int(keep_ratio * (length - len(prompt_tokens)))

            # 6. Decide which tokens to re-mask based on lowest confidence
            # We find the indices of the `num_to_mask` tokens with the lowest probability
            if num_to_mask > 0:
                _, indices_to_mask = torch.topk(sampled_probs, k=num_to_mask, largest=False)
            else:
                indices_to_mask = torch.tensor([], dtype=torch.long, device=tokens.device)
            
            # 7. Update the sequence for the next iteration
            # - Keep the prompt tokens
            # - Keep the high-confidence sampled tokens
            # - Re-mask the low-confidence tokens
            new_tokens = tokens.clone()
            new_tokens[0, :] = torch.where(prompt_mask.squeeze(), sampled_tokens.squeeze(), tokens.squeeze())
            if num_to_mask > 0:
                new_tokens[0, indices_to_mask] = MASK_TOKEN_ID
            tokens = new_tokens

            # Print intermediate result
            print(f"--- Iteration {step + 1}/{steps} (Re-masking {num_to_mask} tokens) ---")
            decoded_text = tokenizer.decode(tokens.squeeze().tolist())
            print(decoded_text.replace(tokenizer.decode([MASK_TOKEN_ID]), "[_]") + "\n")
            
    return tokenizer.decode(tokens.squeeze().tolist())


########################################################################################################
#
# Main execution block
#
########################################################################################################

if not HAS_TRITON:
    print("Cannot run the model without Triton. Exiting.")
    exit()

if WORLD_MODE:
    # Placeholder for WORLD_MODE tokenizer
    pass
else:
    from tokenizers import Tokenizer
    class RWKV_TOKENIZER():
        def __init__(self):
            # Download tokenizer from: https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/20B_tokenizer.json
            tokenizer_path = "../RWKV-v4neo/20B_tokenizer.json"
            if not os.path.exists(tokenizer_path):
                print(f"Tokenizer file not found at {tokenizer_path}")
                print("Please download it from Hugging Face.")
                exit()
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        def encode(self, x):
            return self.tokenizer.encode(x).ids
        def decode(self, x):
            return self.tokenizer.decode(x)
    tokenizer = RWKV_TOKENIZER()

########################################################################################################

print(f'\nUsing CUDA {str(DTYPE).replace("torch.","")}. Loading {args.MODEL_NAME} ...')
model = RWKV_BiDir_x080(args).to('cuda')

final_text = generate_parallel(
    model, 
    tokenizer, 
    prompt, 
    length=GENERATION_LENGTH, 
    steps=GENERATION_STEPS, 
    temp=TEMPERATURE, 
    top_p=TOP_P
)

print("\n--- Final Generated Text ---")
print(final_text)
print('\n')

print("--- How to Train This Model ---")
print("This model is trained as a Masked Language Model (MLM), similar to BERT.")
print("1. Input: A sequence with a certain percentage of tokens randomly replaced by [MASK].")
print("2. Target: The original, unmasked sequence.")
print("3. Loss: Cross-entropy loss is calculated only on the predictions for the masked positions.")
print("This forces the model to use the bidirectional context to infer the missing tokens.")

