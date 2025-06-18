
#!/usr/bin/env python3
"""
Ollama VRAM Benchmark Tool

Benchmarks Mistral-7B-Instruct-v0.3 (Q4_K_M) to find maximum context size
that fits in VRAM before spillover to CPU memory.
"""

import json
import time
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
import requests
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    print("Warning: nvidia-ml-py not available. GPU monitoring disabled.")


class OllamaVRAMBenchmark:
    def __init__(self):
        self.model_name = "huggingface.co/lmstudio-community/mistral-7b-instruct-v0.3-gguf:Q4_K_M"
        self.ollama_url = "http://localhost:11434"
        self.results = []
        self.gpu_handle = None
        self.nvidia_available = NVIDIA_AVAILABLE
        
        if self.nvidia_available:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                print("GPU monitoring initialized")
            except Exception as e:
                print(f"GPU monitoring failed: {e}")
                self.nvidia_available = False

    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def start_ollama_service(self) -> bool:
        """Start Ollama service if not running"""
        if self.check_ollama_service():
            print("Ollama service is already running")
            return True
        
        print("Starting Ollama service...")
        try:
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for _ in range(30):
                if self.check_ollama_service():
                    print("Ollama service started successfully")
                    return True
                time.sleep(1)
            
            print("Failed to start Ollama service")
            return False
        except Exception as e:
            print(f"Error starting Ollama service: {e}")
            return False

    def download_model(self) -> bool:
        """Download the Mistral model if not already available"""
        print(f"Checking if model {self.model_name} is available...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = response.json().get("models", [])
            
            for model in models:
                if self.model_name in model.get("name", ""):
                    print(f"✅ Model {self.model_name} is already available")
                    return True
            
            print(f"📥 Downloading model {self.model_name}...")
            print("This may take several minutes (downloading ~4.1GB)...")
            pull_data = {"name": self.model_name}
            response = requests.post(f"{self.ollama_url}/api/pull", 
                                   json=pull_data, stream=True)
            
            last_status = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Handle different types of progress messages
                        if "status" in data:
                            status = data["status"]
                            
                            # Show progress for downloading
                            if "completed" in data and "total" in data:
                                completed = data["completed"]
                                total = data["total"]
                                percent = (completed / total) * 100 if total > 0 else 0
                                mb_completed = completed / (1024 * 1024)
                                mb_total = total / (1024 * 1024)
                                
                                progress_bar = "█" * int(percent // 2) + "░" * (50 - int(percent // 2))
                                print(f"\r[{progress_bar}] {percent:.1f}% ({mb_completed:.1f}/{mb_total:.1f} MB)", end="", flush=True)
                            
                            # Show status changes
                            elif status != last_status:
                                if last_status:  # Add newline after progress bar
                                    print()
                                print(f"📋 {status}")
                                last_status = status
                            
                            # Check for completion
                            if status == "success":
                                print("\n✅ Model downloaded successfully!")
                                return True
                                
                    except json.JSONDecodeError:
                        continue
            
            print("\n❌ Download completed but no success status received")
            return False
            
        except Exception as e:
            print(f"\n❌ Error downloading model: {e}")
            return False

    def get_gpu_memory_info(self) -> Tuple[int, int]:
        """Get GPU memory usage in MB"""
        if not self.nvidia_available or not self.gpu_handle:
            return 0, 0
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_mb = mem_info.used // (1024 * 1024)
            total_mb = mem_info.total // (1024 * 1024)
            return used_mb, total_mb
        except:
            return 0, 0

    def generate_text(self, context_size: int, num_tokens: int = 100) -> Dict:
        """Generate text with specified context size and measure performance"""
        
        # Create a prompt that will result in the desired context size
        base_prompt = "You are a helpful AI assistant. Please provide detailed explanations."
        filler_text = "This is additional context text to increase the context window size. " * (context_size // 80)
        prompt = base_prompt + "\n\n" + filler_text + "\n\nPlease write a short story about artificial intelligence."
        
        # Measure memory before generation
        gpu_mem_before, gpu_total = self.get_gpu_memory_info()
        cpu_mem_before = psutil.Process().memory_info().rss // (1024 * 1024)
        
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_tokens,
                "temperature": 0.7,
                "num_ctx": context_size
            }
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=request_data, timeout=300)
            response.raise_for_status()
            
            end_time = time.time()
            result = response.json()
            
            # Measure memory after generation
            gpu_mem_after, _ = self.get_gpu_memory_info()
            cpu_mem_after = psutil.Process().memory_info().rss // (1024 * 1024)
            
            generation_time = end_time - start_time
            tokens_generated = len(result.get("response", "").split())
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            return {
                "context_size": context_size,
                "success": True,
                "generation_time": generation_time,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "gpu_mem_before": gpu_mem_before,
                "gpu_mem_after": gpu_mem_after,
                "gpu_mem_used": gpu_mem_after - gpu_mem_before,
                "gpu_total": gpu_total,
                "cpu_mem_before": cpu_mem_before,
                "cpu_mem_after": cpu_mem_after,
                "cpu_mem_used": cpu_mem_after - cpu_mem_before,
                "response_length": len(result.get("response", "")),
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0)
            }
            
        except Exception as e:
            return {
                "context_size": context_size,
                "success": False,
                "error": str(e),
                "gpu_mem_before": gpu_mem_before,
                "gpu_total": gpu_total
            }

    def run_benchmark(self, 
                     start_context: int = 2048, 
                     max_context: int = 32768, 
                     step_size: int = 2048) -> List[Dict]:
        """Run benchmark across different context sizes"""
        
        context_sizes = list(range(start_context, max_context + 1, step_size))
        total_tests = len(context_sizes)
        
        print("🚀 Starting VRAM benchmark...")
        print(f"📊 Context size range: {start_context:,} to {max_context:,} (step: {step_size:,})")
        print(f"🔢 Total tests to run: {total_tests}")
        print("=" * 60)
        
        for i, context_size in enumerate(context_sizes, 1):
            print(f"\n🔍 Test {i}/{total_tests}: Context size {context_size:,}")
            print(f"⏳ Generating text with {context_size:,} token context...")
            
            result = self.generate_text(context_size)
            self.results.append(result)
            
            if result["success"]:
                # Show success with detailed metrics
                print(f"  ✅ SUCCESS")
                print(f"  ⚡ Performance: {result['tokens_per_second']:.1f} tokens/sec")
                print(f"  🖥️  VRAM Usage: {result['gpu_mem_after']:,}/{result['gpu_total']:,} MB ({result['gpu_mem_after']/result['gpu_total']*100:.1f}%)")
                print(f"  ⏱️  Generation time: {result['generation_time']:.2f}s")
                print(f"  📝 Tokens generated: {result['tokens_generated']}")
                
                # Show progress bar
                progress = i / total_tests * 100
                bar_length = 30
                filled_length = int(bar_length * i // total_tests)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                print(f"  📈 Progress: [{bar}] {progress:.1f}% ({i}/{total_tests})")
                
            else:
                print(f"  ❌ FAILED: {result.get('error', 'Unknown error')}")
                print(f"  🛑 Stopping benchmark - likely reached VRAM limit")
                break
            
            # Small delay between tests
            if i < total_tests:
                print(f"  ⏸️  Waiting 2 seconds before next test...")
                time.sleep(2)
        
        print("\n" + "=" * 60)
        print(f"🏁 Benchmark completed! Ran {len(self.results)} tests.")
        
        return self.results

    def analyze_results(self) -> Dict:
        """Analyze benchmark results to find optimal context size"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful benchmarks"}
        
        # Find performance degradation point
        baseline_tps = successful_results[0]["tokens_per_second"]
        degradation_threshold = baseline_tps * 0.8  # 20% performance drop
        
        optimal_context = successful_results[0]["context_size"]
        max_vram_context = successful_results[-1]["context_size"]
        
        for result in successful_results:
            if result["tokens_per_second"] < degradation_threshold:
                break
            optimal_context = result["context_size"]
        
        analysis = {
            "max_successful_context": max_vram_context,
            "optimal_context_size": optimal_context,
            "baseline_performance": baseline_tps,
            "degradation_threshold": degradation_threshold,
            "max_gpu_memory_used": max([r["gpu_mem_after"] for r in successful_results]),
            "total_gpu_memory": successful_results[0]["gpu_total"],
            "total_tests": len(self.results),
            "successful_tests": len(successful_results)
        }
        
        return analysis

    def save_results(self, filename: str = None):
        """Save results to JSON and CSV files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vram_benchmark_{timestamp}"
        
        # Save detailed results as JSON
        with open(f"{filename}.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(f"{filename}.csv", index=False)
        
        # Save analysis
        analysis = self.analyze_results()
        with open(f"{filename}_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Results saved to {filename}.json, {filename}.csv, and {filename}_analysis.json")

    def plot_results(self, filename: str = None):
        """Create visualization of benchmark results"""
        if not self.results:
            return
        
        successful_results = [r for r in self.results if r["success"]]
        if not successful_results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        contexts = [r["context_size"] for r in successful_results]
        tps = [r["tokens_per_second"] for r in successful_results]
        gpu_mem = [r["gpu_mem_after"] for r in successful_results]
        gen_time = [r["generation_time"] for r in successful_results]
        
        # Tokens per second vs context size
        ax1.plot(contexts, tps, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel("Context Size")
        ax1.set_ylabel("Tokens/Second")
        ax1.set_title("Performance vs Context Size")
        ax1.grid(True, alpha=0.3)
        
        # GPU memory usage vs context size
        ax2.plot(contexts, gpu_mem, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel("Context Size")
        ax2.set_ylabel("GPU Memory (MB)")
        ax2.set_title("VRAM Usage vs Context Size")
        ax2.grid(True, alpha=0.3)
        
        # Generation time vs context size
        ax3.plot(contexts, gen_time, 'g-o', linewidth=2, markersize=6)
        ax3.set_xlabel("Context Size")
        ax3.set_ylabel("Generation Time (s)")
        ax3.set_title("Generation Time vs Context Size")
        ax3.grid(True, alpha=0.3)
        
        # Performance efficiency (tokens/sec per MB VRAM)
        efficiency = [t/g for t, g in zip(tps, gpu_mem) if g > 0]
        ax4.plot(contexts[:len(efficiency)], efficiency, 'm-o', linewidth=2, markersize=6)
        ax4.set_xlabel("Context Size")
        ax4.set_ylabel("Efficiency (Tokens/sec per MB)")
        ax4.set_title("VRAM Efficiency vs Context Size")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vram_benchmark_plot_{timestamp}.png"
        else:
            filename = f"{filename}_plot.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")


def main():
    benchmark = OllamaVRAMBenchmark()
    
    # Start Ollama service
    if not benchmark.start_ollama_service():
        print("Failed to start Ollama service. Please start it manually with 'ollama serve'")
        sys.exit(1)
    
    # Download model
    if not benchmark.download_model():
        print("Failed to download model")
        sys.exit(1)
    
    # Run benchmark
    print("\n" + "="*60)
    print("STARTING VRAM BENCHMARK")
    print("="*60)
    
    try:
        results = benchmark.run_benchmark(
            start_context=2048,
            max_context=32768,
            step_size=2048
        )
        
        print("\n" + "="*60)
        print("BENCHMARK ANALYSIS")
        print("="*60)
        
        analysis = benchmark.analyze_results()
        
        if "error" not in analysis:
            print(f"Maximum successful context size: {analysis['max_successful_context']}")
            print(f"Optimal context size (>80% performance): {analysis['optimal_context_size']}")
            print(f"Baseline performance: {analysis['baseline_performance']:.2f} tokens/sec")
            print(f"Maximum VRAM used: {analysis['max_gpu_memory_used']}/{analysis['total_gpu_memory']} MB")
            print(f"VRAM utilization: {(analysis['max_gpu_memory_used']/analysis['total_gpu_memory']*100):.1f}%")
        
        # Save results
        benchmark.save_results()
        benchmark.plot_results()
        
        print("\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if benchmark.results:
            benchmark.save_results()
            benchmark.plot_results()
    except Exception as e:
        print(f"Error during benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

