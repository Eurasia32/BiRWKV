# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BiRWKV-LLADA is a PyTorch implementation of a diffusion language model that combines bidirectional RWKV (Recurrent Weighted Key-Value) attention with LLADA's diffusion framework. The project implements a novel architecture that uses BiWKV attention and BiShift mechanisms for time-conditioned text generation through iterative denoising. This is a research-oriented implementation focused on developing state-of-the-art diffusion language models.

## Key Architecture Components

### Core Files
- `birwkv_modeling.py`: Main BiRWKV-LLADA diffusion model implementation
- `birwkv_train.py`: Training script with diffusion loss and masked language modeling
- `birwkv_test.py`: Inference script with iterative denoising text generation
- `cuda/rwkv8/`: Custom CUDA kernels for high-performance BiRWKV-LLADA attention with time conditioning

### Model Architecture
- **BiRWKVLLADAModel**: Main diffusion model class with time embedding and noise scheduling
- **BiRWKVLLADABlock**: Core attention block with time-conditioned BiWKV and BiShift
- **BiRWKVLLADAAttention**: Enhanced attention mechanism with time conditioning support
- **BiShiftLLADA**: Time-conditioned bidirectional shifting for local dependencies
- **DiffusionScheduler**: Noise scheduling for training and sampling (linear, cosine, sigmoid)
- **TimestepEmbedding**: Sinusoidal time embeddings for diffusion conditioning

### Diffusion Components
- **Time Conditioning**: All major components support timestep-based conditioning
- **Noise Prediction**: Multiple parameterizations (epsilon, v-prediction, sample)
- **Enhanced Sampling**: DDPM-style iterative denoising with guidance support
- **Bidirectional Processing**: Forward and backward RNN states with time modulation

### CUDA Implementation
The project includes enhanced CUDA kernels for diffusion operations:
- `birwkv_op.cpp`: C++ binding for PyTorch with time embedding support
- `birwkv_kernel.cu`: CUDA implementation of time-conditioned bidirectional WKV computation
- **Time Conditioning**: Kernels support time embedding inputs for dynamic parameter modulation
- **Enhanced Stability**: Improved numerical stability with time-dependent clamping
- **Gradient Support**: Full backward pass implementation for time-conditioned operations
- Kernels are dynamically compiled using `torch.utils.cpp_extension.load`

## Development Commands

### Training
```bash
# Start diffusion training from scratch with BiRWKV-LLADA
python birwkv_train.py --batch_size 8 --learning_rate 3e-4 --max_steps 50000 --diffusion_steps 1000

# Resume from checkpoint
python birwkv_train.py --resume_from_checkpoint ./birwkv-llada-model/checkpoint-10000

# Train with specific diffusion parameterization
python birwkv_train.py --prediction_type v_prediction --beta_schedule cosine --token_loss_weight 0.1

# Mixed training mode (diffusion + token prediction)
python birwkv_train.py --diffusion_mode mixed --token_loss_weight 0.2

# Pure diffusion training mode
python birwkv_train.py --diffusion_mode diffusion --diffusion_steps 1000 --prediction_type epsilon

# Smaller model for testing
python birwkv_train.py --d_model 512 --n_layers 12 --batch_size 16 --seq_len 256
```

### Testing/Inference
```bash
# Generate text using diffusion sampling
python birwkv_test.py --checkpoint_path ./birwkv-llada-model/checkpoint-10000/checkpoint.pt --prompt "Your prompt here" --num_steps 20

# Use different sampling strategies
python birwkv_test.py --checkpoint_path model.pt --prompt "Text" --guidance_scale 2.0 --temperature 0.8
```

### Dependencies
The project requires:
- PyTorch with CUDA support (1.12+)
- transformers
- datasets
- sentencepiece
- ninja (for CUDA compilation)
- NVIDIA CUDA Toolkit (11.0+)
- Additional: numpy, tqdm for training utilities

## Important Implementation Details

### Diffusion Process
- **Time Conditioning**: All major components (attention, BiShift, MLP) are conditioned on diffusion timesteps
- **Noise Scheduling**: Supports linear, cosine, and sigmoid beta schedules for training
- **Multi-Parameterization**: Epsilon prediction, v-prediction, and direct sample prediction
- **Enhanced Sampling**: DDPM-style iterative denoising with optional guidance
- **Gradient Stability**: Improved numerical stability through time-dependent parameter modulation

### CUDA Kernel Compilation
- Enhanced kernels with time embedding support are compiled at runtime
- Falls back to PyTorch implementation if CUDA compilation fails
- Requires proper CUDA toolkit installation and C++ compiler
- Extended workspace allocation for diffusion operations (6x larger than base BiRWKV)

### Training Strategy
- **Diffusion Loss**: MSE loss on noise/velocity/sample predictions based on parameterization
- **Time Conditioning**: Random timestep sampling during training
- **Mixed Training**: Optional token prediction loss for better language modeling capabilities
- **Bidirectional Processing**: Enhanced forward/backward RNN states with time modulation
- Supports gradient accumulation and mixed precision training
- Includes checkpoint saving/loading with optimizer and scheduler states

### Model Configuration
- Default model size: 1024 hidden dimensions, 24 layers
- Configurable through `ModelConfig` and `DiffusionConfig` classes
- **Time Embedding**: 128-dimensional sinusoidal embeddings with MLP projection
- **Diffusion Steps**: Default 1000 training steps, configurable inference steps
- Uses RMS Layer Normalization for enhanced stability
- Implements weight tying between embedding and output layers
- **Enhanced BiShift**: Multi-directional time-conditioned shifting patterns

## File Structure Context

```
BiRWKV-LLADA/
├── birwkv_modeling.py    # Enhanced BiRWKV-LLADA diffusion model
├── birwkv_train.py       # Training script with diffusion loss
├── birwkv_test.py        # Diffusion sampling and inference
├── cuda/rwkv8/           # Enhanced CUDA kernels
│   ├── birwkv_op.cpp     # C++ binding with time embedding support
│   └── birwkv_kernel.cu  # Time-conditioned CUDA implementation
├── Paper/Ref/            # Research papers
├── CLAUDE.md             # This guidance file
└── README.md             # Basic project info
```

## Development Notes

- The codebase implements a novel BiRWKV-LLADA diffusion architecture
- Model uses custom autograd functions for CUDA kernel integration with time conditioning
- Implements both forward and backward passes for the enhanced BiRWKV attention mechanism
- **Diffusion Features**: Time embeddings, noise scheduling, multiple prediction parameterizations
- **Enhanced Stability**: Time-dependent numerical clamping and parameter modulation
- Uses streaming datasets for memory efficiency during training
- Includes comprehensive error handling for CUDA compilation failures
- **Time Conditioning**: All major components support timestep-based parameter modulation
- **Sampling Methods**: DDPM-style iterative denoising with configurable guidance
- Contains mixed English and Chinese comments reflecting research development process