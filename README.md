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
- Multiple iterations for statistical accuracy
- Conversation mode for realistic multi-turn benchmarking
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

### Quick Start
```bash
python run_benchmark.py --quick
```

### Full Benchmark
```bash
python run_benchmark.py --model mistral:7b-instruct-q4_K_M
```

### Conversation Mode
For more realistic benchmarking with multi-turn dialogues:
```bash
# Run with conversation mode (3 turns per test)
python run_benchmark.py --conversation

# Custom number of conversation turns
python run_benchmark.py --conversation --turns 5
```

### Advanced Options
```bash
# Custom context range and iterations
python run_benchmark.py \
    --model mistral:7b-instruct-q4_K_M \
    --start 2048 \
    --max 16384 \
    --step 2048 \
    --iterations 3

# Conversation mode with custom settings
python run_benchmark.py \
    --conversation \
    --turns 4 \
    --iterations 5
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
