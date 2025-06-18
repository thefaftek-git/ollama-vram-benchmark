# Ollama VRAM Benchmark Tool

This tool benchmarks the Mistral-7B-Instruct-v0.3 model (Q4_K_M quantization) to determine the maximum context size that can fit in VRAM before overflow to CPU memory.

## Hardware Target
- GPU: RTX 5060 Ti with 16GB VRAM
- Expected baseline: ~70 tokens/second at 4096 context window (from LMStudio)

## Features
- Automated model download and setup
- Progressive context size testing
- VRAM usage monitoring
- Performance metrics collection
- Automatic detection of VRAM overflow point

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ollama service:
```bash
ollama serve &
```

3. Run benchmark:
```bash
python benchmark.py
```

## Output
The tool generates a detailed report showing:
- Maximum context size achievable in VRAM
- Performance metrics at different context sizes
- VRAM usage patterns
- Recommended optimal context size

## Model Details
- Model: lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF
- Quantization: Q4_K_M
- Base model size: ~4.1GB (Q4_K_M quantization)
