import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("  Memory Allocated:", torch.cuda.memory_allocated(i) / 1024**2, "MB")
        print("  Memory Reserved: ", torch.cuda.memory_reserved(i) / 1024**2, "MB")
        print("  Total Memory:    ", torch.cuda.get_device_properties(i).total_memory / 1024**2, "MB")
else:
    print("No CUDA device is available")
