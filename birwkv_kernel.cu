#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cmath>

// Utility to get tensor pointers
template<typename T>
T* get_ptr(torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

// CUDA kernel for the forward pass of Bi-WKV
template<typename T_in>
__global__ void bi_wkv_forward_kernel(
    const T_in* r, const T_in* k, const T_in* v, const float* w, const float* u,
    T_in* y,
    // Workspace to store intermediate states for the backward pass
    float* num_fwd, float* den_fwd, float* num_bwd, float* den_bwd,
    const int B, const int seq_len, const int C)
{
    const int b = blockIdx.x;
    const int c = threadIdx.x + blockIdx.y * blockDim.x;
    if (c >= C) return;

    const int offset = b * seq_len * C + c;
    
    // Time decay factor. w is learnable, so we use -exp(w) to keep it negative.
    const float decay = expf(-expf(w[c]));

    // Forward RNN (left-to-right)
    float num_state = 0.0f; // numerator state
    float den_state = 0.0f; // denominator state
    for (int t = 0; t < seq_len; ++t) {
        const int idx = offset + t * C;
        const float kt = static_cast<float>(k[idx]);
        const float vt = static_cast<float>(v[idx]);
        const float exp_kt = expf(kt);
        
        num_state = num_state * decay + exp_kt * vt;
        den_state = den_state * decay + exp_kt;
        num_fwd[idx] = num_state;
        den_fwd[idx] = den_state;
    }

    // Backward RNN (right-to-left)
    num_state = 0.0f;
    den_state = 0.0f;
    for (int t = seq_len - 1; t >= 0; --t) {
        const int idx = offset + t * C;
        const float kt = static_cast<float>(k[idx]);
        const float vt = static_cast<float>(v[idx]);
        const float exp_kt = expf(kt);

        num_state = num_state * decay + exp_kt * vt;
        den_state = den_state * decay + exp_kt;
        num_bwd[idx] = num_state;
        den_bwd[idx] = den_state;
    }
    
    // Final combination
    for (int t = 0; t < seq_len; ++t) {
        const int idx = offset + t * C;
        const float kt = static_cast<float>(k[idx]);
        const float vt = static_cast<float>(v[idx]);
        const float rt = static_cast<float>(r[idx]);
        
        const float exp_kt = expf(kt);
        const float bonus_num = expf(u[c] + kt) * vt;
        const float bonus_den = expf(u[c] + kt);
        
        // num = fwd_num + bwd_num - current_contribution + bonus_contribution
        const float num = num_fwd[idx] + num_bwd[idx] - (exp_kt * vt) + bonus_num;
        // den = fwd_den + bwd_den - current_contribution + bonus_contribution
        const float den = den_fwd[idx] + den_bwd[idx] - exp_kt + bonus_den;
        
        const float wkv = num / fmaxf(den, 1e-8f);
        const float sig_r = 1.0f / (1.0f + expf(-rt));
        
        y[idx] = static_cast<T_in>(sig_r * wkv);
    }
}


// CUDA kernel for the backward pass of Bi-WKV (Full BPTT implementation)
template<typename T_in>
__global__ void bi_wkv_backward_kernel(
    const T_in* r, const T_in* k, const T_in* v, const float* w, const float* u,
    const T_in* grad_y,
    const float* num_fwd, const float* den_fwd, const float* num_bwd, const float* den_bwd,
    T_in* grad_r, T_in* grad_k, T_in* grad_v, float* grad_w, float* grad_u,
    const int B, const int seq_len, const int C)
{
    const int b = blockIdx.x;
    const int c = threadIdx.x + blockIdx.y * blockDim.x;
    if (c >= C) return;

    const int offset = b * seq_len * C + c;
    
    const float decay = expf(-expf(w[c]));
    const float d_decay_d_w = -expf(w[c]) * decay;

    float grad_w_acc = 0.0f;
    float grad_u_acc = 0.0f;

    // --- Pass 1: Backward in time (for forward states) ---
    float ga_fwd = 0.0f; // gradient w.r.t. numerator state
    float gb_fwd = 0.0f; // gradient w.r.t. denominator state
    for (int t = seq_len - 1; t >= 0; --t) {
        const int idx = offset + t * C;
        
        const float kt = static_cast<float>(k[idx]);
        const float vt = static_cast<float>(v[idx]);
        const float rt = static_cast<float>(r[idx]);
        const float sig_r = 1.0f / (1.0f + expf(-rt));

        const float exp_kt = expf(kt);
        const float bonus_num = expf(u[c] + kt) * vt;
        const float bonus_den = expf(u[c] + kt);
        const float num = num_fwd[idx] + num_bwd[idx] - (exp_kt * vt) + bonus_num;
        const float den = den_fwd[idx] + den_bwd[idx] - exp_kt + bonus_den;
        const float wkv = num / fmaxf(den, 1e-8f);

        const float grad_wkv = static_cast<float>(grad_y[idx]) * sig_r;
        const float grad_num = grad_wkv / den;
        const float grad_den = -grad_wkv * wkv / den;

        const float ga_fwd_prop = ga_fwd * decay;
        const float gb_fwd_prop = gb_fwd * decay;

        const float total_ga = grad_num + ga_fwd_prop;
        const float total_gb = grad_den + gb_fwd_prop;
        
        const float exp_uk = expf(u[c] + kt);
        
        grad_r[idx] = static_cast<T_in>(static_cast<float>(grad_y[idx]) * wkv * sig_r * (1.0f - sig_r));
        grad_v[idx] = static_cast<T_in>(total_ga * exp_kt + grad_num * (exp_uk - exp_kt));
        grad_k[idx] = static_cast<T_in>(
            total_ga * exp_kt * vt + total_gb * exp_kt +
            grad_num * (exp_uk * vt - exp_kt * vt) +
            grad_den * (exp_uk - exp_kt)
        );
        
        grad_u_acc += grad_num * exp_uk * vt + grad_den * exp_uk;

        if (t > 0) {
            grad_w_acc += (ga_fwd * num_fwd[idx - C] + gb_fwd * den_fwd[idx - C]) * d_decay_d_w;
        }

        ga_fwd = total_ga;
        gb_fwd = total_gb;
    }

    // --- Pass 2: Forward in time (for backward states) ---
    float ga_bwd = 0.0f;
    float gb_bwd = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        const int idx = offset + t * C;
        
        const float kt = static_cast<float>(k[idx]);
        const float vt = static_cast<float>(v[idx]);
        const float rt = static_cast<float>(r[idx]);
        const float sig_r = 1.0f / (1.0f + expf(-rt));

        const float exp_kt = expf(kt);
        const float bonus_num = expf(u[c] + kt) * vt;
        const float bonus_den = expf(u[c] + kt);
        const float num = num_fwd[idx] + num_bwd[idx] - (exp_kt * vt) + bonus_num;
        const float den = den_fwd[idx] + den_bwd[idx] - exp_kt + bonus_den;
        const float wkv = num / fmaxf(den, 1e-8f);

        const float grad_wkv = static_cast<float>(grad_y[idx]) * sig_r;
        const float grad_num = grad_wkv / den;
        const float grad_den = -grad_wkv * wkv / den;

        const float ga_bwd_prop = ga_bwd * decay;
        const float gb_bwd_prop = gb_bwd * decay;
        
        const float total_ga = grad_num + ga_bwd_prop;
        const float total_gb = grad_den + gb_bwd_prop;
        
        grad_v[idx] += static_cast<T_in>(total_ga * exp_kt);
        grad_k[idx] += static_cast<T_in>(total_ga * exp_kt * vt + total_gb * exp_kt);
        
        if (t < seq_len - 1) {
            grad_w_acc += (ga_bwd * num_bwd[idx + C] + gb_bwd * den_bwd[idx + C]) * d_decay_d_w;
        }

        ga_bwd = total_ga;
        gb_bwd = total_gb;
    }
    
    atomicAdd(grad_w + c, grad_w_acc);
    atomicAdd(grad_u + c, grad_u_acc);
}


// --- Kernel Launchers ---

void bi_wkv_forward(
    torch::Tensor r, torch::Tensor k, torch::Tensor v, torch::Tensor w, torch::Tensor u,
    torch::Tensor y, torch::Tensor workspace)
{
    const int B = r.size(0);
    const int T = r.size(1);
    const int C = r.size(2);

    const int threads = 256;
    const dim3 blocks(B, (C + threads - 1) / threads);

    float* num_fwd = reinterpret_cast<float*>(workspace.data_ptr());
    float* den_fwd = num_fwd + B * T * C;
    float* num_bwd = den_fwd + B * T * C;
    float* den_bwd = num_bwd + B * T * C;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(r.scalar_type(), "bi_wkv_forward", ([&] {
        bi_wkv_forward_kernel<scalar_t><<<blocks, threads>>>(
            get_ptr<scalar_t>(r), get_ptr<scalar_t>(k), get_ptr<scalar_t>(v),
            get_ptr<float>(w), get_ptr<float>(u),
            get_ptr<scalar_t>(y),
            num_fwd, den_fwd, num_bwd, den_bwd,
            B, T, C
        );
    }));
}

void bi_wkv_backward(
    torch::Tensor r, torch::Tensor k, torch::Tensor v, torch::Tensor w, torch::Tensor u, torch::Tensor y,
    torch::Tensor grad_y, torch::Tensor workspace,
    torch::Tensor grad_r, torch::Tensor grad_k, torch::Tensor grad_v, torch::Tensor grad_w, torch::Tensor grad_u)
{
    const int B = r.size(0);
    const int T = r.size(1);
    const int C = r.size(2);

    const int threads = 256;
    const dim3 blocks(B, (C + threads - 1) / threads);

    const float* num_fwd = reinterpret_cast<const float*>(workspace.data_ptr());
    const float* den_fwd = num_fwd + B * T * C;
    const float* num_bwd = den_fwd + B * T * C;
    const float* den_bwd = num_bwd + B * T * C;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(r.scalar_type(), "bi_wkv_backward", ([&] {
        bi_wkv_backward_kernel<scalar_t><<<blocks, threads>>>(
            get_ptr<scalar_t>(r), get_ptr<scalar_t>(k), get_ptr<scalar_t>(v),
            get_ptr<float>(w), get_ptr<float>(u),
            get_ptr<scalar_t>(grad_y),
            num_fwd, den_fwd, num_bwd, den_bwd,
            get_ptr<scalar_t>(grad_r), get_ptr<scalar_t>(grad_k), get_ptr<scalar_t>(grad_v),
            get_ptr<float>(grad_w), get_ptr<float>(grad_u),
            B, T, C
        );
    }));
}
