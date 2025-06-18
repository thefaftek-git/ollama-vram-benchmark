
"""
Configuration file for VRAM benchmark settings
"""

# Model configuration
MODEL_CONFIG = {
    "name": "mistral:7b-instruct-v0.3-q4_K_M",
    "expected_size_gb": 4.1,  # Q4_K_M quantization size
    "baseline_tps": 70,  # Expected tokens/second from LMStudio
    "baseline_context": 4096
}

# Benchmark configuration
BENCHMARK_CONFIG = {
    "start_context": 2048,
    "max_context": 32768,
    "step_size": 2048,
    "num_tokens_generate": 100,
    "temperature": 0.7,
    "delay_between_tests": 2,  # seconds
    "timeout": 300,  # seconds per test
    "performance_threshold": 0.8  # 80% of baseline for "optimal" detection
}

# Hardware configuration (RTX 5060 Ti)
HARDWARE_CONFIG = {
    "gpu_name": "RTX 5060 Ti",
    "vram_gb": 16,
    "expected_usable_vram_gb": 15.5,  # Account for OS/driver overhead
    "cuda_cores": 4352,
    "memory_bus": 128  # bit
}

# Output configuration
OUTPUT_CONFIG = {
    "save_json": True,
    "save_csv": True,
    "save_plots": True,
    "plot_format": "png",
    "plot_dpi": 300
}

