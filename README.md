# Cheet Sheet:

## Look at GPU usage: 
-     watch -n 1 gpustat
-     watch -n 1 nvidia-smi


## Trace the GPU peak memory
    Reset memory statistics:   
      torch.cuda.reset_peak_memory_stats()

    Current and peak memory usage
        current_mem = torch.cuda.memory_allocated() / 1024**2  # In MB
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # In MB

        print(f"Current GPU memory usage: {current_mem:.2f} MB")
        print(f"Peak GPU memory usage: {peak_mem:.2f} MB")


    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"Total GPU memory: {info.total / 1024**2:.2f} MB")
    print(f"Used GPU memory: {info.used / 1024**2:.2f} MB")
    print(f"Free GPU memory: {info.free / 1024**2:.2f} MB")