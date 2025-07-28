import torch
import torch.nn as nn
import torch.nn.functional as F  # Added import for F.softmax
import math  # Added import for math.sqrt
from MLA.MLA import MultiHeadLatentAttention


def test_multi_head_latent_attention():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    

    # 16B MOdel parameters
    # {
    # "vocab_size": 102400,
    # "dim": 2048,
    # "inter_dim": 10944,
    # "moe_inter_dim": 1408,
    # "n_layers": 27,
    # "n_dense_layers": 1,
    # "n_heads": 16,
    # "n_routed_experts": 64,
    # "n_shared_experts": 2,
    # "n_activated_experts": 6,
    # "route_scale": 1.0,
    # "q_lora_rank": 0,
    # "kv_lora_rank": 512,
    # "qk_nope_head_dim": 128,
    # "qk_rope_head_dim": 64,
    # "v_head_dim": 128,
    # "mscale": 0.707
    # }






    # Hyperparameters
    batch_size = 1
    seq_len = 10
    hidden_dim = 2048
    num_heads = 16
    head_dim = 128
    kv_compression_dim = 128
    query_compression_dim = 128
    rope_dim = 64
    dropout_rate = 0.1
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"Input shape: {x.shape}")
    
    # Create sample attention mask (all ones, no padding)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Initialize the model
    model = MultiHeadLatentAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_compression_dim=kv_compression_dim,
        query_compression_dim=query_compression_dim,
        rope_dim=rope_dim,
        dropout_rate=dropout_rate
    )
    
    # Forward pass with shape tracking
    print("\nStep 1: Key/Value Compression")
    kv_compressed = model.kv_down(x)
    print(f"  Before kv_down: {x.shape}")
    print(f"  After kv_down: {kv_compressed.shape}")
    
    print("\nStep 2: Key/Value Projection")
    keys_c = model.key_up(kv_compressed).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    values = model.value_up(kv_compressed).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    keys_r = model.key_rope(kv_compressed).view(batch_size, seq_len, num_heads, rope_dim).transpose(1, 2)
    keys_r = model.rope(keys_r)
    print(f"  After key_up (keys_c): {keys_c.shape}")
    print(f"  After value_up (values): {values.shape}")
    print(f"  After key_rope (keys_r): {keys_r.shape}")
    
    print("\nStep 3: Query Compression")
    query_compressed = model.query_down(x)
    print(f"  Before query_down: {x.shape}")
    print(f"  After query_down: {query_compressed.shape}")
    
    print("\nStep 4: Query Projection")
    queries_c = model.query_up(query_compressed).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    queries_r = model.query_rope(query_compressed).view(batch_size, seq_len, num_heads, rope_dim).transpose(1, 2)
    queries_r = model.rope(queries_r)
    print(f"  After query_up (queries_c): {queries_c.shape}")
    print(f"  After query_rope (queries_r): {queries_r.shape}")
    
    print("\nStep 5: Concatenate Queries and Keys")
    queries = torch.cat([queries_c, queries_r], dim=-1)
    keys = torch.cat([keys_c, keys_r], dim=-1)
    print(f"  After concatenation (queries): {queries.shape}")
    print(f"  After concatenation (keys): {keys.shape}")
    
    print("\nStep 6: Compute Attention Scores")
    attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(head_dim + rope_dim)
    print(f"  Attention scores shape: {attn_scores.shape}")
    
    print("\nStep 7: Apply Causal and Attention Masks")
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
            attention_mask = ~attention_mask
        combined_mask = causal_mask | attention_mask
    else:
        combined_mask = causal_mask
    attn_scores = attn_scores.masked_fill(combined_mask, float("-1e9"))
    print(f"  After masking (attn_scores): {attn_scores.shape}")
    
    print("\nStep 8: Compute Attention Probabilities")
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_probs = model.dropout(attn_probs)
    print(f"  Attention probabilities shape: {attn_probs.shape}")
    
    print("\nStep 9: Apply Attention to Values")
    context = torch.matmul(attn_probs, values)
    print(f"  Context shape after matmul: {context.shape}")
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    print(f"  Context shape after reshape: {context.shape}")
    
    print("\nStep 10: Output Projection")
    output = model.output_proj(context)
    output = model.dropout(output)
    print(f"  Before output_proj: {context.shape}")
    print(f"  After output_proj: {output.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    print(f"\nOutput shape matches expected: {output.shape}")

if __name__ == "__main__":
    test_multi_head_latent_attention()