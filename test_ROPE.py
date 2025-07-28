import torch
import torch.nn as nn
from ROPE.ROPE import RotaryPositionalEmbedding

def test_rotary_positional_embedding():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters (matching MultiHeadLatentAttention usage)
    batch_size = 1
    num_heads = 16
    seq_len = 10
    rope_dim = 64
    
    # Create sample input tensor
    x = torch.randn(batch_size, num_heads, seq_len, rope_dim)
    print(f"Input shape: {x.shape}")
    
    # Initialize the RoPE module
    model = RotaryPositionalEmbedding(dim=rope_dim)
    
    # Forward pass with shape tracking
    print("\nStep 1: Apply Rotary Positional Embeddings")
    output = model(x)
    print(f"  Before RoPE: {x.shape}")
    print(f"  After RoPE: {output.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, num_heads, seq_len, rope_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    print(f"\nOutput shape matches expected: {output.shape}")

    # Optional: Check that the output is not identical to input (RoPE modifies values)
    assert not torch.allclose(x, output), "Output should differ from input due to RoPE transformation"
    print("RoPE transformation applied successfully (output differs from input)")

if __name__ == "__main__":
    test_rotary_positional_embedding()