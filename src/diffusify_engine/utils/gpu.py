import torch
from typing import List, Optional, Union

def get_available_gpus() -> List[str]:
    """Get list of available CUDA devices."""
    num_gpus = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(num_gpus)]

def parse_gpu_devices(devices_str: Optional[str] = None) -> List[str]:
    """Parse GPU devices string into list of device identifiers.

    Args:
        devices_str: Comma-separated string of GPU devices
        (e.g., "cuda:0,cuda:1") If None, returns all available GPUs
    
    Returns:
        List of device identifiers
    """
    if not devices_str:
        return get_available_gpus()
    
    devices = [d.strip() for d in devices_str.split(",")]
    available = get_available_gpus()
    
    # Validate devices
    for device in devices:
        if device not in available:
            raise ValueError(f"Invalid device {device}. Available devices: {available}")
    
    return devices

def get_gpu_memory(device: Union[str, torch.device]) -> int:
    """Get available memory for specified GPU device."""
    if isinstance(device, str):
        device = torch.device(device)
    return torch.cuda.get_device_properties(device).total_memory

def print_gpu_info(devices: List[str]) -> None:
    """Print information about specified GPU devices."""
    print("\nGPU Information:")
    print("-" * 50)
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        memory_gb = props.total_memory / (1024**3)
        print(f"Device: {device}")
        print(f"Name: {props.name}")
        print(f"Memory: {memory_gb:.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print("-" * 50)
