#!/usr/bin/env python3
"""
Test platform detection and GPU monitoring initialization
"""

import platform
import subprocess
import psutil

# Detect platform
PLATFORM = platform.system().lower()
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"
IS_WINDOWS = PLATFORM == "windows"

print(f"🖥️  Platform Detection:")
print(f"   System: {platform.system()}")
print(f"   Machine: {platform.machine()}")
print(f"   Platform: {PLATFORM}")
print(f"   Is macOS: {IS_MACOS}")
print(f"   Is Linux: {IS_LINUX}")
print(f"   Is Windows: {IS_WINDOWS}")

# Apple Silicon detection
IS_APPLE_SILICON = False
if IS_MACOS:
    print(f"\n🍎 macOS Detection:")
    try:
        # Check if running on Apple Silicon
        cpu_brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        print(f"   CPU Brand: {cpu_brand}")
        IS_APPLE_SILICON = 'Apple' in cpu_brand
        print(f"   Apple Silicon (brand check): {IS_APPLE_SILICON}")
    except Exception as e:
        print(f"   CPU brand check failed: {e}")
        # Alternative check
        try:
            machine = platform.machine().lower()
            print(f"   Machine architecture: {machine}")
            IS_APPLE_SILICON = machine in ['arm64', 'aarch64']
            print(f"   Apple Silicon (machine check): {IS_APPLE_SILICON}")
        except Exception as e2:
            print(f"   Machine check failed: {e2}")
            IS_APPLE_SILICON = False

print(f"\n🧠 Final Apple Silicon Detection: {IS_APPLE_SILICON}")

# Test GPU monitoring availability
print(f"\n💾 Memory Monitoring Test:")

# NVIDIA check
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle).decode('utf-8')
    print(f"   ✅ NVIDIA GPU available: {gpu_name}")
    nvidia_available = True
except Exception as e:
    print(f"   ❌ NVIDIA GPU not available: {e}")
    nvidia_available = False

# Determine what would be used
if nvidia_available:
    print(f"   → Would use: NVIDIA GPU monitoring")
elif IS_APPLE_SILICON:
    print(f"   → Would use: Apple Silicon unified memory monitoring")
elif IS_MACOS:
    print(f"   → Would use: Intel Mac system memory monitoring")
else:
    print(f"   → Would use: Generic system memory monitoring")

# Test memory reading
print(f"\n📊 Memory Reading Test:")
try:
    memory = psutil.virtual_memory()
    total_mb = memory.total // (1024 * 1024)
    used_mb = memory.used // (1024 * 1024)
    print(f"   Total Memory: {total_mb:,} MB")
    print(f"   Used Memory: {used_mb:,} MB")
    print(f"   Memory Usage: {(used_mb/total_mb)*100:.1f}%")
    
    if IS_APPLE_SILICON:
        print(f"\n🧪 Apple Silicon vm_stat test:")
        try:
            result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:10]  # First 10 lines
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
                print("   ✅ vm_stat working")
            else:
                print(f"   ❌ vm_stat failed with return code: {result.returncode}")
        except Exception as e:
            print(f"   ❌ vm_stat error: {e}")
            
except Exception as e:
    print(f"   ❌ Memory reading failed: {e}")

print(f"\n🎯 Summary:")
print(f"   Platform: {PLATFORM}")
print(f"   Apple Silicon: {IS_APPLE_SILICON}")
print(f"   NVIDIA Available: {nvidia_available}")
gpu_monitoring_available = nvidia_available or IS_APPLE_SILICON or IS_MACOS
print(f"   GPU Monitoring Available: {gpu_monitoring_available}")

if gpu_monitoring_available:
    print(f"   ✅ This platform should work with the benchmark!")
else:
    print(f"   ⚠️  This platform has limited monitoring capabilities")
