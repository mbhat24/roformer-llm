"""
Hardware detection and optimization for RoFormer LLM
"""

import torch
import psutil
import os
import platform


def detect_hardware():
    """Detect available hardware and return optimal configuration"""
    print("="*60)
    print("HARDWARE DETECTION")
    print("="*60)
    
    # System info
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    print(f"\nCPU: {cpu_physical} physical cores, {cpu_count} logical threads")
    
    # Memory info
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.total / 1024**3:.1f} GB total, {mem.available / 1024**3:.1f} GB available")
    
    # GPU detection
    device_info = {
        'device': 'cpu',
        'device_name': 'CPU',
        'memory_gb': mem.available / 1024**3,
        'recommended_batch_size': 16,
        'recommended_max_length': 256,
        'recommended_embed_dim': 256,
        'recommended_num_layers': 4,
        'recommended_num_heads': 4
    }
    
    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\n✓ CUDA (NVIDIA) available: {device_count} device(s)")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        
        device_info['device'] = 'cuda'
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_info['recommended_batch_size'] = 32
        device_info['recommended_max_length'] = 512
        device_info['recommended_embed_dim'] = 512
        device_info['recommended_num_layers'] = 8
        device_info['recommended_num_heads'] = 8
    
    # MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        print(f"\n✓ MPS (Apple Silicon) available")
        print(f"  Metal Performance Shaders enabled")
        
        device_info['device'] = 'mps'
        device_info['device_name'] = 'Apple Silicon GPU (MPS)'
        device_info['memory_gb'] = mem.available / 1024**3  # Shared memory
        device_info['recommended_batch_size'] = 16
        device_info['recommended_max_length'] = 512
        device_info['recommended_embed_dim'] = 384
        device_info['recommended_num_layers'] = 6
        device_info['recommended_num_heads'] = 6
    
    # CPU only
    else:
        print(f"\n✗ No GPU detected, using CPU")
        print(f"  Training will be slower but functional")
    
    # Optimize based on available memory
    if device_info['memory_gb'] < 8:
        print(f"\n⚠ Low memory detected (< 8 GB)")
        print(f"  Using conservative settings")
        device_info['recommended_batch_size'] = 8
        device_info['recommended_embed_dim'] = 256
        device_info['recommended_num_layers'] = 4
    elif device_info['memory_gb'] >= 16:
        print(f"\n✓ High memory available (≥ 16 GB)")
        print(f"  Can use larger batch sizes")
        device_info['recommended_batch_size'] = 32
        device_info['recommended_embed_dim'] = 512
        device_info['recommended_num_layers'] = 8
    
    print(f"\n{'='*60}")
    print("RECOMMENDED CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device_info['device']} ({device_info['device_name']})")
    print(f"Batch size: {device_info['recommended_batch_size']}")
    print(f"Max sequence length: {device_info['recommended_max_length']}")
    print(f"Embedding dimension: {device_info['recommended_embed_dim']}")
    print(f"Number of layers: {device_info['recommended_num_layers']}")
    print(f"Number of heads: {device_info['recommended_num_heads']}")
    print(f"{'='*60}\n")
    
    return device_info


def get_optimal_device():
    """Get the optimal device for training"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


if __name__ == "__main__":
    hardware_info = detect_hardware()
    device = get_optimal_device()
    print(f"Optimal device: {device}")
