#include <cuda_bf16.h>
#include <assert.h>

// 定义 bfloat16 类型别名和转换函数
using bf = __nv_bfloat16;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

typedef bf * __restrict__ F_;

/**
 * @brief 双向WKV前向传播CUDA Kernel
 *
 * 该Kernel实现了WKV计算的双向版本。它通过两个独立的pass来完成：
 * 1. 前向pass (从左到右): 计算一个标准的单向WKV，并将结果存入输出张量y_。
 * 2. 后向pass (从右到左): 计算另一个单向WKV，但沿时间维度反向进行，然后将其结果与前向pass的结果相加。
 *
 * 为了支持后向传播，它会保存两个方向的状态检查点(s_fwd_, s_bwd_)。
 */
__global__ void bi_wkv7_forward_kernel(int T, int H,
                                  F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                  bf* y_,
                                  float* s_fwd_, float* sa_fwd_,
                                  float* s_bwd_, float* sa_bwd_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    __shared__ float q[C], k[C], w[C], a[C], b[C];

    // --- 1. 前向pass: 从左到右 (t = 0 -> T-1) ---
    float state_fwd[C] = {0};
    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        // 加载当前时间步的输入
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        // 计算 sa (state attention)
        float sa = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state_fwd[j];
        }
        // 保存 sa_fwd 以便后向传播使用
        if (sa_fwd_ != nullptr) sa_fwd_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0;
        // 更新状态并计算输出
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state_fwd[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = to_bf(y); // 将前向pass的输出写入最终结果

        // 保存状态检查点
        if (s_fwd_ != nullptr && (t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
            #pragma unroll
            for (int j = 0; j < C; j++) {
                s_fwd_[base + j*C] = state_fwd[j];
            }
        }
    }

    // --- 2. 后向pass: 从右到左 (t = T-1 -> 0) ---
    float state_bwd[C] = {0};
    for (int t = T - 1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        // 重新加载当前时间步的输入
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        // 计算 sa
        float sa = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state_bwd[j];
        }
        // 保存 sa_bwd
        if (sa_bwd_ != nullptr) sa_bwd_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0;
        // 更新状态并计算输出
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state_bwd[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        // 将后向pass的输出与已有的前向结果相加
        y_[ind] = to_bf(to_float(y_[ind]) + y);

        // 保存状态检查点 (注意t的计算方式)
        if (s_bwd_ != nullptr && (T-t)%_CHUNK_LEN_ == 0) {
            int chunk_idx = (T-1-t)/_CHUNK_LEN_;
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + chunk_idx*C*C + i;
            #pragma unroll
            for (int j = 0; j < C; j++) {
                s_bwd_[base + j*C] = state_bwd[j];
            }
        }
    }
}


/**
 * @brief 双向WKV后向传播CUDA Kernel
 *
 * 该Kernel为双向WKV计算梯度。由于最终输出 y = y_fwd + y_bwd，
 * 因此梯度 dy 对于两个路径是相同的。
 *
 * 1. 计算L->R路径的梯度: 从 t = T-1 到 0 进行标准的后向传播。
 * 2. 计算R->L路径的梯度: 从 t = 0 到 T-1 进行对称的后向传播。
 *
 * 两个路径计算出的梯度会通过 atomicAdd 累加到最终的梯度张量中。
 */
__global__ void bi_wkv7_backward_kernel(int T, int H,
                                   F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_,
                                   float * __restrict__ s_fwd_, float * __restrict__ sa_fwd_,
                                   float * __restrict__ s_bwd_, float * __restrict__ sa_bwd_,
                                   bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C];
    __shared__ float sa_fwd[C], sa_bwd[C];
    __shared__ float dSb_fwd_shared[C], dSb_bwd_shared[C];

    // --- 1. L->R 路径的后向传播 (t = T-1 -> 0) ---
    float stateT_fwd[C] = {0}, dstate_fwd[C] = {0}, dstateT_fwd[C] = {0};
    for (int t = T - 1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        // 加载输入和梯度
        float qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        float wi = __expf(wi_fac);
        q[i] = qi; w[i] = wi;
        k[i] = to_float(k_[ind]); v[i] = to_float(v_[ind]);
        a[i] = to_float(a_[ind]); b[i] = to_float(b_[ind]);
        dy[i] = to_float(dy_[ind]); sa_fwd[i] = sa_fwd_[ind];
        __syncthreads();

        // 加载状态检查点
        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
            for (int j = 0; j < C; j++) stateT_fwd[j] = s_fwd_[base + j];
        }

        // 计算梯度
        float dq = 0; for (int j = 0; j < C; j++) dq += stateT_fwd[j]*dy[j];
        float iwi = 1.0f/wi;
        for (int j = 0; j < C; j++) {
            stateT_fwd[j] = (stateT_fwd[j] - k[i]*v[j] - b[i]*sa_fwd[j]) * iwi;
            dstate_fwd[j] += dy[i] * q[j];
            dstateT_fwd[j] += qi * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
        for (int j = 0; j < C; j++) {
            dw += dstateT_fwd[j]*stateT_fwd[j]; dk += dstateT_fwd[j]*v[j];
            dv += dstate_fwd[j]*k[j]; dSb += dstate_fwd[j]*b[j]; db += dstateT_fwd[j]*sa_fwd[j];
        }
        atomicAdd(&((float*)dw_)[ind], dw * wi * wi_fac);
        atomicAdd(&((float*)dk_)[ind], dk);
        atomicAdd(&((float*)dv_)[ind], dv);
        atomicAdd(&((float*)db_)[ind], db);

        __syncthreads(); dSb_fwd_shared[i] = dSb; __syncthreads();
        float da = 0; for (int j = 0; j < C; j++) da += stateT_fwd[j]*dSb_fwd_shared[j];
        atomicAdd(&((float*)da_)[ind], da);

        for (int j = 0; j < C; j++) {
            dstate_fwd[j] = dstate_fwd[j]*w[j] + dSb * a[j];
            dstateT_fwd[j] = dstateT_fwd[j]*wi + a[i] * dSb_fwd_shared[j];
        }
    }

    // --- 2. R->L 路径的后向传播 (t = 0 -> T-1) ---
    // 这个过程与L->R的后向传播对称，但沿时间正向进行
    float stateT_bwd[C] = {0}, dstate_bwd[C] = {0}, dstateT_bwd[C] = {0};
    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        // 加载输入和梯度
        float qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        float wi = __expf(wi_fac);
        q[i] = qi; w[i] = wi;
        k[i] = to_float(k_[ind]); v[i] = to_float(v_[ind]);
        a[i] = to_float(a_[ind]); b[i] = to_float(b_[ind]);
        dy[i] = to_float(dy_[ind]); sa_bwd[i] = sa_bwd_[ind];
        __syncthreads();

        // 加载状态检查点 (注意t的计算方式)
        if ((T-t)%_CHUNK_LEN_ == 0) {
            int chunk_idx = (T-1-t)/_CHUNK_LEN_;
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + chunk_idx*C*C + i*C;
            for (int j = 0; j < C; j++) stateT_bwd[j] = s_bwd_[base + j];
        }

        // 计算梯度
        float dq = 0; for (int j = 0; j < C; j++) dq += stateT_bwd[j]*dy[j];
        float iwi = 1.0f/wi;
        for (int j = 0; j < C; j++) {
            stateT_bwd[j] = (stateT_bwd[j] - k[i]*v[j] - b[i]*sa_bwd[j]) * iwi;
            dstate_bwd[j] += dy[i] * q[j];
            dstateT_bwd[j] += qi * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
        for (int j = 0; j < C; j++) {
            dw += dstateT_bwd[j]*stateT_bwd[j]; dk += dstateT_bwd[j]*v[j];
            dv += dstate_bwd[j]*k[j]; dSb += dstate_bwd[j]*b[j]; db += dstateT_bwd[j]*sa_bwd[j];
        }
        // 使用 atomicAdd 将梯度累加到最终结果
        atomicAdd(&((float*)dw_)[ind], dw * wi * wi_fac);
        atomicAdd(&((float*)dk_)[ind], dk);
        atomicAdd(&((float*)dv_)[ind], dv);
        atomicAdd(&((float*)db_)[ind], db);
        atomicAdd(&((float*)dq_)[ind], dq); // dq也需要累加

        __syncthreads(); dSb_bwd_shared[i] = dSb; __syncthreads();
        float da = 0; for (int j = 0; j < C; j++) da += stateT_bwd[j]*dSb_bwd_shared[j];
        atomicAdd(&((float*)da_)[ind], da);

        // 更新 dstate，为下一个时间步做准备
        for (int j = 0; j < C; j++) {
            dstate_bwd[j] = dstate_bwd[j]*w[j] + dSb * a[j];
            dstateT_bwd[j] = dstateT_bwd[j]*wi + a[i] * dSb_bwd_shared[j];
        }
    }
}

// --- 核函数启动器 ---

void bi_wkv7_forward(
    torch::Tensor r, torch::Tensor k, torch::Tensor v, torch::Tensor w, torch::Tensor u,
    torch::Tensor y, torch::Tensor workspace)
{
    const int B = r.size(0);
    const int seq_len = r.size(1);
    const int C = r.size(2);

    const int threads = 256;
    const dim3 blocks(B, (C + threads - 1) / threads);

    float* num_fwd = reinterpret_cast<float*>(workspace.data_ptr());
    float* den_fwd = num_fwd + B * seq_len * C;
    float* num_bwd = den_fwd + B * seq_len * C;
    float* den_bwd = num_bwd + B * seq_len * C;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r.scalar_type(), "bi_wkv_forward_launcher", ([&] {
        bi_wkv7_forward_kernel<scalar_t><<<blocks, threads>>>(
            get_ptr<scalar_t>(r), get_ptr<scalar_t>(k), get_ptr<scalar_t>(v),
            get_ptr<float>(w), get_ptr<float>(u),
            get_ptr<scalar_t>(y),
            num_fwd, den_fwd, num_bwd, den_bwd,
            B, seq_len, C
        );
    }));
}

void bi_wkv7_backward(
    torch::Tensor r, torch::Tensor k, torch::Tensor v, torch::Tensor w, torch::Tensor u, torch::Tensor y,
    torch::Tensor grad_y, torch::Tensor workspace,
    torch::Tensor grad_r, torch::Tensor grad_k, torch::Tensor grad_v, torch::Tensor grad_w, torch::Tensor grad_u)
{
    const int B = r.size(0);
    const int seq_len = r.size(1);
    const int C = r.size(2);

    const int threads = 256;
    const dim3 blocks(B, (C + threads - 1) / threads);

    const float* num_fwd = reinterpret_cast<const float*>(workspace.data_ptr());
    const float* den_fwd = num_fwd + B * seq_len * C;
    const float* num_bwd = den_fwd + B * seq_len * C;
    const float* den_bwd = num_bwd + B * seq_len * C;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r.scalar_type(), "bi_wkv_backward_launcher", ([&] {
        bi_wkv7_backward_kernel<scalar_t><<<blocks, threads>>>(
            get_ptr<scalar_t>(r), get_ptr<scalar_t>(k), get_ptr<scalar_t>(v),
            get_ptr<float>(w), get_ptr<float>(u),
            get_ptr<scalar_t>(grad_y),
            num_fwd, den_fwd, num_bwd, den_bwd,
            get_ptr<scalar_t>(grad_r), get_ptr<scalar_t>(grad_k), get_ptr<scalar_t>(grad_v),
            get_ptr<float>(grad_w), get_ptr<float>(grad_u),
            B, seq_len, C
        );
    }));
}