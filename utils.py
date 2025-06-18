

"""
Utility functions for the VRAM benchmark tool
"""

import subprocess
import psutil
import time
import requests
from typing import Dict, List, Optional, Tuple


def check_system_requirements() -> Dict[str, bool]:
    """Check if system meets requirements for benchmarking"""
    requirements = {
        "ollama_installed": False,
        "nvidia_gpu": False,
        "sufficient_ram": False,
        "python_packages": False
    }
    
    # Check Ollama installation
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        requirements["ollama_installed"] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # Check NVIDIA GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        requirements["nvidia_gpu"] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # Check system RAM (recommend at least 8GB for safety)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    requirements["sufficient_ram"] = total_ram_gb >= 8
    
    # Check Python packages
    try:
        import requests
        import pandas
        import matplotlib
        requirements["python_packages"] = True
    except ImportError:
        pass
    
    return requirements


def get_gpu_info() -> Dict:
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = [p.strip() for p in line.split(',')]
            
            return {
                "name": parts[0],
                "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "memory_free_mb": int(parts[3]),
                "utilization_percent": int(parts[4])
            }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    
    return {}


def wait_for_ollama_service(url: str = "http://localhost:11434", timeout: int = 30) -> bool:
    """Wait for Ollama service to become available"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    
    return False


def estimate_context_memory_usage(context_size: int, model_size_gb: float = 4.1) -> Dict[str, float]:
    """Estimate memory usage for given context size"""
    
    # Rough estimates based on transformer architecture
    # These are approximations and may vary based on implementation
    
    # KV cache grows linearly with context size
    # For Mistral-7B, roughly 2 bytes per token per layer for KV cache
    num_layers = 32  # Mistral-7B has 32 layers
    bytes_per_token_kv = 2 * num_layers  # Key + Value
    
    kv_cache_mb = (context_size * bytes_per_token_kv) / (1024 * 1024)
    
    # Model weights (relatively constant)
    model_mb = model_size_gb * 1024
    
    # Additional overhead (attention matrices, activations, etc.)
    overhead_mb = context_size * 0.1  # Rough estimate
    
    total_estimated_mb = model_mb + kv_cache_mb + overhead_mb
    
    return {
        "model_mb": model_mb,
        "kv_cache_mb": kv_cache_mb,
        "overhead_mb": overhead_mb,
        "total_estimated_mb": total_estimated_mb,
        "context_size": context_size
    }


def generate_test_prompt(target_length: int) -> str:
    """Generate a test prompt of approximately target length"""
    
    base_prompt = """You are a helpful AI assistant. I need you to help me understand artificial intelligence and machine learning concepts. Please provide detailed explanations when answering questions.

Context: Artificial intelligence has been developing rapidly in recent years, with significant advances in natural language processing, computer vision, and machine learning algorithms. Large language models like GPT, Claude, and others have shown remarkable capabilities in understanding and generating human-like text.

"""
    
    # Add filler text to reach target length
    filler_unit = "This is additional context text designed to increase the total context window size for testing purposes. It contains various topics and information to simulate a realistic conversation history. "
    
    current_length = len(base_prompt)
    remaining_length = target_length - current_length - 200  # Leave room for the actual question
    
    if remaining_length > 0:
        num_units = remaining_length // len(filler_unit)
        filler_text = filler_unit * num_units
        
        # Add some variety to the filler
        topics = [
            "Machine learning algorithms include supervised, unsupervised, and reinforcement learning approaches. ",
            "Neural networks consist of interconnected nodes that process information in layers. ",
            "Deep learning has revolutionized fields like computer vision and natural language processing. ",
            "Transformer architectures have become the foundation for many modern AI systems. ",
            "Training large models requires significant computational resources and careful optimization. "
        ]
        
        for i in range(0, num_units, 5):
            filler_text = filler_text.replace(filler_unit, topics[i % len(topics)], 1)
    else:
        filler_text = ""
    
    question = "\n\nGiven this context, please write a detailed explanation about how attention mechanisms work in transformer models. Include technical details and examples."
    
    return base_prompt + filler_text + question


def format_bytes(bytes_val: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def print_system_info():
    """Print system information for debugging"""
    print("System Information:")
    print("-" * 40)
    
    # CPU info
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {format_bytes(psutil.virtual_memory().total)}")
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"VRAM: {gpu_info['memory_total_mb']} MB")
        print(f"VRAM Used: {gpu_info['memory_used_mb']} MB")
        print(f"GPU Utilization: {gpu_info['utilization_percent']}%")
    else:
        print("GPU: Not detected or nvidia-smi not available")
    
    # Requirements check
    print("\nRequirements Check:")
    print("-" * 40)
    requirements = check_system_requirements()
    for req, status in requirements.items():
        status_str = "✓" if status else "✗"
        print(f"{status_str} {req.replace('_', ' ').title()}")
    
    print()


