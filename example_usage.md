# Example Usage

## Basic Usage Examples

### Quick Test
```bash
# Run a quick test to verify everything works
python run_benchmark.py --quick
```

### Standard Benchmark
```bash
# Run full benchmark with single prompt/response mode
python run_benchmark.py --model mistral:7b-instruct-q4_K_M
```

### Conversation Mode (Recommended)
```bash
# Run with conversation mode for more realistic benchmarking
python run_benchmark.py --conversation --model mistral:7b-instruct-q4_K_M
```

## Advanced Examples

### High-Accuracy Benchmarking
```bash
# More iterations for better statistical accuracy
python run_benchmark.py \
    --conversation \
    --turns 4 \
    --iterations 10 \
    --model mistral:7b-instruct-q4_K_M
```

### Testing Specific Context Ranges
```bash
# Test only higher context sizes
python run_benchmark.py \
    --start 8192 \
    --max 32768 \
    --step 4096 \
    --conversation \
    --turns 3
```

### Custom Model Testing
```bash
# Test a different model
python run_benchmark.py \
    --model llama2:7b-chat-q4_K_M \
    --conversation \
    --turns 3
```

## Understanding the Results

### Key Metrics
- **Tokens/Second**: Generation speed (higher is better)
- **VRAM Usage**: Memory consumption in MB
- **Context Size**: Input context length in tokens
- **Performance Margin**: 5% performance threshold analysis

### Conversation Mode Benefits
- **More Realistic**: Simulates actual multi-turn usage
- **Better Context Testing**: Tests how performance degrades with growing conversation history
- **Real-world Patterns**: Captures memory usage patterns of actual conversations

### Output Interpretation
- Look for the context size where tokens/second drops significantly
- Monitor VRAM usage to find the overflow point
- Use statistical averages from multiple iterations for reliability
