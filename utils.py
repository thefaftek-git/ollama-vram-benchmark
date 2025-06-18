

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


def generate_test_prompt(target_context_size: int) -> str:
    """Generate a more natural test prompt that targets specific context size"""
    
    base_prompt = """You are a helpful AI assistant. I need you to help me understand artificial intelligence and machine learning concepts. Please provide detailed explanations when answering questions.

Context: Artificial intelligence has been developing rapidly in recent years, with significant advances in natural language processing, computer vision, and machine learning algorithms. Large language models like GPT, Claude, and others have shown remarkable capabilities in understanding and generating human-like text."""

    # Add realistic technical content instead of simple filler
    technical_content = """

Machine learning encompasses various paradigms including supervised learning where models learn from labeled data, unsupervised learning which discovers patterns in unlabeled data, and reinforcement learning where agents learn through environmental interaction. Deep neural networks, inspired by biological neural systems, consist of interconnected layers of artificial neurons that process information through weighted connections and activation functions.

Transformer architectures have revolutionized natural language processing through their attention mechanisms, which allow models to focus on relevant parts of input sequences. These models use multi-head attention to capture different types of relationships between tokens, enabling superior performance on tasks like translation, summarization, and text generation.

Computer vision has similarly benefited from deep learning advances, with convolutional neural networks excelling at image recognition tasks through hierarchical feature extraction. Residual connections, batch normalization, and attention mechanisms have further improved model performance and training stability.

The training process for large language models involves processing massive datasets containing billions of tokens, requiring sophisticated optimization techniques like gradient clipping, learning rate scheduling, and distributed training across multiple GPUs. Model architectures continue to evolve with innovations in attention mechanisms, positional encoding, and normalization techniques."""

    # Calculate how much additional content we need (rough token estimation: ~4 chars per token)
    current_tokens = len((base_prompt + technical_content)) // 4
    target_tokens = target_context_size - 50  # Leave room for the final question
    
    if current_tokens < target_tokens:
        # Add more technical content to reach target size
        additional_content = technical_content
        while current_tokens < target_tokens:
            additional_content += technical_content
            current_tokens = len((base_prompt + additional_content)) // 4
        
        full_content = base_prompt + additional_content
    else:
        full_content = base_prompt + technical_content
    
    # Add the final question
    final_question = "\n\nGiven this context, please write a detailed explanation about how attention mechanisms work in transformer models. Include technical details and examples."
    
    return full_content + final_question


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


def generate_conversation_starter(target_context_size: int) -> str:
    """Generate a conversation starter that encourages longer responses"""
    
    starters = [
        "I'm planning a complex software project and need advice. The project involves creating a distributed system with microservices, real-time data processing, user authentication, and machine learning components. Can you help me understand the architecture decisions I need to make?",
        
        "I'm writing a research paper on the environmental impact of technology. I need to explore topics like e-waste, energy consumption of data centers, sustainable computing practices, and green technology innovations. What are the key points I should cover?",
        
        "I'm designing a game that combines strategy, role-playing, and simulation elements. The setting is a space colony where players manage resources, develop technology, engage in diplomacy, and explore unknown territories. What mechanics would make this engaging?",
        
        "I'm learning about artificial intelligence and want to understand different approaches. Can you explain machine learning, deep learning, neural networks, natural language processing, computer vision, and how they're applied in real-world scenarios?",
        
        "I'm interested in sustainable living and want to make significant changes. This includes renewable energy, organic gardening, waste reduction, sustainable transportation, ethical consumption, and community building. Where should I start?"
    ]
    
    # Select based on context size for variety
    import hashlib
    hash_val = int(hashlib.md5(str(target_context_size).encode()).hexdigest(), 16)
    starter = starters[hash_val % len(starters)]
    
    return starter


def generate_followup_question(turn_number: int) -> str:
    """Generate follow-up questions for conversation turns"""
    
    questions = [
        "That's very helpful! Can you elaborate on the most important aspects you mentioned? I'd like to understand the practical steps and potential challenges involved.",
        
        "Interesting points! What would you recommend as the best practices or most effective approaches? Are there any common mistakes I should avoid?",
        
        "Thank you for the detailed explanation! How would you prioritize these different elements? What should I focus on first, and how do they interconnect?",
        
        "This gives me a great foundation to work with! Can you provide some specific examples or case studies that illustrate these concepts in action? What have others done successfully?",
        
        "Excellent insights! What tools, resources, or next steps would you recommend? How can I continue learning and implementing these ideas effectively?"
    ]
    
    return questions[turn_number % len(questions)]


