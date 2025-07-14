#include <torch/extension.h>

// Forward declarations of our CUDA functions
void bi_wkv_forward(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                    torch::Tensor w, torch::Tensor u, torch::Tensor y,
                    torch::Tensor workspace);

void bi_wkv_backward(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                     torch::Tensor w, torch::Tensor u, torch::Tensor y,
                     torch::Tensor grad_y, torch::Tensor workspace,
                     torch::Tensor grad_r, torch::Tensor grad_k,
                     torch::Tensor grad_v, torch::Tensor grad_w,
                     torch::Tensor grad_u);

// C++ wrapper for the forward pass
void forward_wrapper(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                     torch::Tensor w, torch::Tensor u, torch::Tensor y,
                     torch::Tensor workspace) {
  TORCH_CHECK(r.is_cuda(), "Input tensors must be on CUDA");
  TORCH_CHECK(k.is_cuda(), "Input tensors must be on CUDA");
  TORCH_CHECK(v.is_cuda(), "Input tensors must be on CUDA");
  bi_wkv_forward(r, k, v, w, u, y, workspace);
}

// C++ wrapper for the backward pass
void backward_wrapper(torch::Tensor r, torch::Tensor k, torch::Tensor v,
                      torch::Tensor w, torch::Tensor u, torch::Tensor y,
                      torch::Tensor grad_y, torch::Tensor workspace,
                      torch::Tensor grad_r, torch::Tensor grad_k,
                      torch::Tensor grad_v, torch::Tensor grad_w,
                      torch::Tensor grad_u) {
  TORCH_CHECK(r.is_cuda(), "Input tensors must be on CUDA");
  bi_wkv_backward(r, k, v, w, u, y, grad_y, workspace, grad_r, grad_k, grad_v,
                  grad_w, grad_u);
}

// Binding to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "Bi-WKV forward (CUDA)");
  m.def("backward", &backward_wrapper, "Bi-WKV backward (CUDA)");
}
