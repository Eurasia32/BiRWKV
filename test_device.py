#!/usr/bin/env python3
"""
Test script to verify device handling in BiRWKV-LLADA model
"""
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from birwkv_modeling import ModelConfig, BiRWKVLLADAModel
    
    print("Testing BiRWKV-LLADA device handling...")
    
    # Test with small model to reduce memory usage
    config = ModelConfig(
        d_model=256,
        n_layers=2,
        vocab_size=1000,
        max_sequence_length=128,
    )
    
    model = BiRWKVLLADAModel(config)
    
    # Test CPU first
    print("Testing CPU...")
    device = torch.device("cpu")
    model.to(device)
    
    # Create test input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    # Test forward pass
    output = model(input_ids)
    print(f"✓ CPU forward pass successful, output shape: {output.logits.shape}")
    
    # Test diffusion loss
    loss_dict = model.compute_diffusion_loss(input_ids)
    print(f"✓ CPU diffusion loss successful, loss: {loss_dict['diffusion_loss']:.4f}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("\nTesting CUDA...")
        device = torch.device("cuda")
        model.to(device)
        input_ids = input_ids.to(device)
        
        # Test forward pass
        output = model(input_ids)
        print(f"✓ CUDA forward pass successful, output shape: {output.logits.shape}")
        
        # Test diffusion loss
        loss_dict = model.compute_diffusion_loss(input_ids)
        print(f"✓ CUDA diffusion loss successful, loss: {loss_dict['diffusion_loss']:.4f}")
    else:
        print("\n⚠️  CUDA not available, skipping CUDA tests")
    
    print("\n🎉 All device handling tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()