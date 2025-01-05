import torch
import time

def compute_sum_broadcasted(a, b, c, device):
    """
    Compute broadcasted sum for given tensors
    Args:
        a, b, c: Input tensors of shape (n_events, dim, 1, 1), (n_events, 1, dim, 1), (n_events, 1, 1, dim)
    Returns:
        Summed and broadcasted result
    """
    a_broadcasted = a.expand(-1, 100, 100, 100).to(device)  # Keep first dim, expand others
    b_broadcasted = b.expand(-1, 100, 100, 100).to(device)
    c_broadcasted = c.expand(-1, 100, 100, 100).to(device)

    result = a_broadcasted**2 + b_broadcasted**2 + c_broadcasted**2 + a_broadcasted*b_broadcasted + b_broadcasted*c_broadcasted + c_broadcasted*a_broadcasted + torch.erf(a_broadcasted) + torch.erf(b_broadcasted) + torch.erf(c_broadcasted)
    for _ in range(100):
        result += a_broadcasted**2 + b_broadcasted**2 + c_broadcasted**2 + a_broadcasted*b_broadcasted + b_broadcasted*c_broadcasted + c_broadcasted*a_broadcasted + torch.erf(a_broadcasted) + torch.erf(b_broadcasted) + torch.erf(c_broadcasted)

    return result

def compute_sum_broadcasted_mask(a, b, c, mask, device):
    """
    Compute broadcasted sum for given tensors
    Args:
        a, b, c: Input tensors of shape (n_events, dim, 1, 1), (n_events, 1, dim, 1), (n_events, 1, 1, dim)
    Returns:
        Summed and broadcasted result
    """
    mask = mask.to(device)
    expanded_mask = mask.unsqueeze(0).expand(a.shape[0], -1, -1, -1)
    
    # Only compute for masked elements
    result = torch.zeros(a.shape[0], 100, 100, 100, device=device)

    # Get masked indices
    batch_indices, x_indices, y_indices, z_indices = torch.where(expanded_mask)
    
    # Index tensors efficiently
    a_masked = a[batch_indices, x_indices, :, :]
    b_masked = b[batch_indices, :, y_indices, :]
    c_masked = c[batch_indices, :, :, z_indices]

    

    masked_result = a_masked**2 + b_masked**2 + c_masked**2 + \
                   a_masked*b_masked + b_masked*c_masked + c_masked*a_masked + \
                   torch.erf(a_masked) + torch.erf(b_masked) + torch.erf(c_masked)
                   
    for _ in range(100):
        masked_result += a_masked**2 + b_masked**2 + c_masked**2 + \
                        a_masked*b_masked + b_masked*c_masked + c_masked*a_masked + \
                        torch.erf(a_masked) + torch.erf(b_masked) + torch.erf(c_masked)
    
    # Place results back in full tensor
    result[batch_indices, x_indices, y_indices, z_indices] = masked_result.squeeze()

    return result

def check_consistency(a, b, c, mask, device):
    """
    Compare results between full calculation with mask applied and masked calculation
    Args:
        result_full: Result tensor from unmasked calculation
        result_masked: Result tensor from masked calculation
        mask: Boolean mask used in masked calculation
    Returns:
        max_diff, min_diff, mean_diff
    """

    result_full = compute_sum_broadcasted(a, b, c, device)
    result_masked = compute_sum_broadcasted_mask(a, b, c, mask, device)

    # Expand mask to match batch dimension
    expanded_mask = mask.unsqueeze(0).expand(result_full.shape[0], -1, -1, -1)
    
    # Get values only where mask is True
    full_masked_values = result_full[expanded_mask]
    masked_values = result_masked[expanded_mask]
    
    # Calculate differences
    diff = (full_masked_values - masked_values).abs()
    max_diff = diff.max().item()
    min_diff = diff.min().item()
    mean_diff = diff.mean().item()
    
    return max_diff, min_diff, mean_diff

if __name__ == "__main__":
    # Example tensors with 100 events
    n_events = 100
    device = 'cuda'
    a = torch.randn(n_events, 100, 1, 1, device=device)    # Shape: (100, 100, 1, 1)
    b = torch.randn(n_events, 1, 100, 1, device=device)    # Shape: (100, 1, 100, 1) 
    c = torch.randn(n_events, 1, 1, 100, device=device)    # Shape: (100, 1, 1, 100)

    print("No Masking") 
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    # Compute result for all events
    result1 = compute_sum_broadcasted(a, b, c, device)  # Shape: (100, 100, 100, 100)
    gpu_time = time.time() - start_time
    current_mem = torch.cuda.memory_allocated() / 1024**2  # In MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # In MB
    print(f"\nCurrent GPU memory usage: {current_mem:.2f} MB")
    print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
    print(f"GPU Time: {gpu_time:.4f} seconds")

    del result1
    torch.cuda.empty_cache()


    # Masking
    print("\nMasking")
    mask = torch.zeros(100, 100, 100, dtype=torch.bool)
    num_ones = int(0.01 * mask.numel())  # 1% of total elements
    indices = torch.randint(0, mask.numel(), (num_ones,))
    mask.view(-1)[indices] = True

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    # Compute result for all events
    result2 = compute_sum_broadcasted_mask(a, b, c, mask, device)  # Shape: (100, 100, 100, 100)
    gpu_time = time.time() - start_time
    current_mem = torch.cuda.memory_allocated() / 1024**2  # In MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # In MB
    print(f"\nCurrent GPU memory usage: {current_mem:.2f} MB")
    print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
    print(f"GPU Time: {gpu_time:.4f} seconds")

    # Check consistency between results
    print("\nChecking consistency:")
    max_diff, min_diff, mean_diff = check_consistency(a, b, c, mask, device)
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Minimum absolute difference: {min_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

