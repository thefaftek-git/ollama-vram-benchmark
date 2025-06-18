# Cross-Platform Usage Guide

## Platform-Specific Features

### NVIDIA GPUs (Linux/Windows)
- Full VRAM monitoring with detailed GPU metrics
- Requires `nvidia-ml-py` package (auto-installed)
- Shows GPU name, temperature, and precise VRAM usage
- Original target platform for RTX 5060 Ti

### Apple Silicon Macs (M1/M2/M3/M4)
- Unified memory architecture support
- Uses `vm_stat` for accurate memory tracking
- Shows "Unified Memory" in all outputs and plots
- Optimized for Apple's memory-on-package design

### Intel Macs
- System memory monitoring fallback
- Uses `psutil` for memory tracking
- Shows "Memory" in outputs

### Other Platforms
- Basic memory monitoring using `psutil`
- Graceful degradation for unsupported systems

## Example Usage

### On NVIDIA GPU System:
```bash
python3 benchmark.py --start-context 2048 --max-context 16384 --step-size 2048
```

### On Apple Silicon Mac:
```bash
python3 benchmark.py --start-context 4096 --max-context 32768 --step-size 4096 --conversation
```

### Quick Test (any platform):
```bash
python3 benchmark.py --start-context 2048 --max-context 8192 --step-size 2048 --iterations 2
```

## Platform Detection

The tool automatically detects:
- Operating system (macOS, Linux, Windows)
- CPU architecture (Apple Silicon vs Intel/AMD)
- Available GPU monitoring capabilities
- Appropriate memory terminology

## Memory Monitoring Methods

| Platform | Method | Memory Type | Accuracy |
|----------|--------|-------------|----------|
| NVIDIA GPU | nvidia-ml-py | VRAM | High |
| Apple Silicon | vm_stat | Unified Memory | High |
| Intel Mac | psutil | System Memory | Medium |
| Other | psutil | System Memory | Medium |

## Output Differences

### NVIDIA:
```
🖥️  Platform: Linux
🎮 GPU: NVIDIA GeForce RTX 5060 Ti
💾 Total VRAM: 16,384 MB
```

### Apple Silicon:
```
🖥️  Platform: Darwin
🧠 Architecture: Apple Silicon (Unified Memory)
💾 Total Unified Memory: 32,768 MB
```

### Intel Mac:
```
🖥️  Platform: Darwin
🧠 Architecture: Intel Mac
💾 Total Memory: 16,384 MB
```
