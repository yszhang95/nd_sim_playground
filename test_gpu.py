import torch
import time
import numpy as np
import psutil
import os
import tracemalloc
#from memory_profiler import profile
from torch.profiler import profile, ProfilerActivity

#@profile
def get_cpu_info():
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'total_cores': psutil.cpu_count(logical=True),
        'cpu_freq': psutil.cpu_freq(),
        'cpu_usage': psutil.cpu_percent(interval=1, percpu=True)
    }
    return cpu_info


def get_gpu_info():
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        'name': torch.cuda.get_device_name(0),
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'max_memory': torch.cuda.get_device_properties(0).total_memory / 1e9,  # GB
        'compute_capability': torch.cuda.get_device_capability(0),
        'multi_processor_count': torch.cuda.get_device_properties(0).multi_processor_count
    }
    return gpu_info


def display_system_resources():
    print("=== System Resource Information ===")
    
    # CPU Information
    cpu_info = get_cpu_info()
    print("\nCPU Information:")
    print(f"Physical cores: {cpu_info['physical_cores']}")
    print(f"Total cores: {cpu_info['total_cores']}")
    print(f"Current frequency: {cpu_info['cpu_freq'].current:.2f} MHz")
    print(f"CPU Usage per core: {cpu_info['cpu_usage']}%")
    
    # GPU Information
    gpu_info = get_gpu_info()
    if gpu_info:
        print("\nGPU Information:")
        print(f"GPU Name: {gpu_info['name']}")
        print(f"Number of GPUs: {gpu_info['device_count']}")
        print(f"Current GPU: {gpu_info['current_device']}")
        print(f"Total Memory: {gpu_info['max_memory']:.2f} GB")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        print(f"Number of SMs: {gpu_info['multi_processor_count']}")
    else:
        print("\nNo GPU available")


def compare_cpu_gpu_operations(size=1000000):
    # Create tensors
    cpu_tensor = torch.randn(size)
    gpu_tensor = cpu_tensor.cuda()  # Move to GPU
    
    # CPU Operation
    start_time = time.time()
    cpu_result = cpu_tensor * 2 + 1
    cpu_time = time.time() - start_time
    
    # GPU Operation
    start_time = time.time()
    gpu_result = gpu_tensor * 2 + 1
    for _ in range(10000):
        gpu_result = gpu_tensor * 2 + 1
    torch.cuda.synchronize()  # Wait for GPU operation to complete
    gpu_time = time.time() - start_time
    
    # Memory Usage
    print(f"GPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    
    # Time Comparison
    print(f"\nTime Comparison:")
    print(f"CPU Time: {cpu_time:.4f} seconds")
    print(f"GPU Time: {gpu_time:.4f} seconds")
    
    # Verify results match
    cpu_numpy = cpu_result.numpy()
    gpu_numpy = gpu_result.cpu().numpy()
    print(f"\nResults match: {np.allclose(cpu_numpy, gpu_numpy)}")


def memory_management_example():
    # Allocate tensor on GPU
    x = torch.randn(1000000, device='cuda')
    
    # Clear memory
    del x
    torch.cuda.empty_cache()
    
    # Check memory after cleanup
    print(f"\nAfter cleanup:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

if __name__ == "__main__":

    # # Start tracing memory allocation
    # tracemalloc.start()

    #display_system_resources()
    #print("\n" + "="*50 + "\n")

    # Reset memory statistics
    torch.cuda.reset_peak_memory_stats()

    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}\n")
        compare_cpu_gpu_operations(100000000)
        memory_management_example()
    else:
        print("No GPU available!")

    # Current and peak memory usage
    current_mem = torch.cuda.memory_allocated() / 1024**2  # In MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # In MB

    print(f"Current GPU memory usage: {current_mem:.2f} MB")
    print(f"Peak GPU memory usage: {peak_mem:.2f} MB")

    # # Get peak memory (CPU) usage
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    # print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

    # # Stop the tracing
    # tracemalloc.stop()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        a = torch.rand((1000, 1000), device='cuda')
        b = torch.rand((1000, 1000), device='cuda')
        c = a @ b

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    #print(prof.key_averages().table(sort_by="cuda_time_total"))