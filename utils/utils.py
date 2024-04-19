import subprocess
from typing import List, Tuple
import torch

def get_gpu_memory() -> List[int]:
    """
    Get the current GPU memory usage for all available GPUs using nvidia-smi.

    Returns:
        List[int]: A list of memory usage values for each GPU in megabytes (MB).
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ], text=True)

    return [int(x) for x in result.strip().split('\n')]

def get_gpus_avail() -> List[Tuple[int, float]]:
    """
    Get the IDs of GPUs where the memory usage is less than or equal to 40% of the total capacity.
    Assumes a fixed total memory per GPU (should ideally be parameterized or calculated).

    Returns:
        List[Tuple[int, float]]: A list of tuples containing GPU IDs and their corresponding memory usage as a percentage.
    """
    total_memory_per_gpu = 11178  # Example value for a specific GPU model, should be parameterized
    memory_usage = get_gpu_memory()
    memory_usage_percent = [usage / total_memory_per_gpu for usage in memory_usage]

    available_gpus = [(idx, usage) for idx, usage in enumerate(memory_usage_percent) if usage <= 0.4]

    print("CUDA ID  Memory Usage")
    if available_gpus:
        for idx, mem in available_gpus:
            print(f"{idx:^7}  {mem * 100:>10.2f}%")
    else:
        print("No GPUs available under 40% usage")
        for idx, mem in enumerate(memory_usage_percent):
            print(f"{idx:^7}  {mem * 100:>10.2f}%")

    return sorted(available_gpus, key=lambda x: (x[1], -x[0]))

def select_device(device: str = "gpu") -> torch.device:
    """
    Select the most appropriate device based on the specified preference and availability.

    Parameters:
        device (str): Preferred device type ('gpu' or 'cpu'). If 'gpu' is preferred but not available, falls back to 'cpu'.

    Returns:
        torch.device: The selected device.
    """
    if device == "cpu":
        return torch.device("cpu")
    elif device == "gpu":
        available_gpus = get_gpus_avail()
        if available_gpus:
            return torch.device(f"cuda:{available_gpus[0][0]}")
        else:
            print("Falling back to CPU as no suitable GPU is available.")
            return torch.device("cpu")

if __name__ == "__main__":
    # Example usage:
    print("Selected device:", select_device("cpu"))
    print("Selected device:", select_device("gpu"))
