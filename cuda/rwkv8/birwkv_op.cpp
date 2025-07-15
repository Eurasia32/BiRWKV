#include <torch/extension.h>

// Forward declarations of our enhanced CUDA functions for BiRWKV-LLADA
void bi_wkv_llada_forward(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                         torch::Tensor w, torch::Tensor u, torch::Tensor time_emb,
                         torch::Tensor y, torch::Tensor workspace);

void bi_wkv_llada_backward(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                          torch::Tensor w, torch::Tensor u, torch::Tensor time_emb,
                          torch::Tensor y, torch::Tensor grad_y, torch::Tensor workspace,
                          torch::Tensor grad_r, torch::Tensor grad_k,
                          torch::Tensor grad_v, torch::Tensor grad_w,
                          torch::Tensor grad_u, torch::Tensor grad_time_emb);

// C++ wrapper for the forward pass
void forward_wrapper(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                     torch::Tensor w, torch::Tensor u, torch::Tensor time_emb,
                     torch::Tensor y, torch::Tensor workspace) {
  TORCH_CHECK(r.is_cuda(), "Input tensors must be on CUDA");
  TORCH_CHECK(k.is_cuda(), "Input tensors must be on CUDA");
  TORCH_CHECK(v.is_cuda(), "Input tensors must be on CUDA");
  TORCH_CHECK(time_emb.is_cuda(), "Time embedding must be on CUDA");
  bi_wkv_llada_forward(r, k, v, w, u, time_emb, y, workspace);
}

// C++ wrapper for the backward pass
void backward_wrapper(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                      torch::Tensor w, torch::Tensor u, torch::Tensor time_emb,
                      torch::Tensor y, torch::Tensor grad_y, torch::Tensor workspace,
                      torch::Tensor grad_r, torch::Tensor grad_k,
                      torch::Tensor grad_v, torch::Tensor grad_w,
                      torch::Tensor grad_u, torch::Tensor grad_time_emb) {
  TORCH_CHECK(r.is_cuda(), "Input tensors must be on CUDA");
  bi_wkv_llada_backward(r, k, v, w, u, time_emb, y, grad_y, workspace,
                       grad_r, grad_k, grad_v, grad_w, grad_u, grad_time_emb);
}

// Binding to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "Bi-WKV-LLADA forward (CUDA)");
  m.def("backward", &backward_wrapper, "Bi-WKV-LLADA backward (CUDA)");
}
