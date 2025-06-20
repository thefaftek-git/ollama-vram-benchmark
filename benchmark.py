
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
import ollama
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import platform

# GPU monitoring imports
try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

# Detect platform
PLATFORM = platform.system().lower()
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"
IS_WINDOWS = PLATFORM == "windows"

# Apple Silicon detection
IS_APPLE_SILICON = False
if IS_MACOS:
    try:
        # Check if running on Apple Silicon
        cpu_brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        IS_APPLE_SILICON = 'Apple' in cpu_brand
    except:
        # Alternative check
        try:
            machine = platform.machine().lower()
            IS_APPLE_SILICON = machine in ['arm64', 'aarch64']
        except:
            IS_APPLE_SILICON = False


class OllamaVRAMBenchmark:
    def __init__(self):
        self.model_name = "huggingface.co/lmstudio-community/mistral-7b-instruct-v0.3-gguf:Q4_K_M"
        self.ollama_url = "http://localhost:11434"
        self.results = []
        self.gpu_handle = None
        self.nvidia_available = NVIDIA_AVAILABLE
        self.gpu_monitoring_available = False
        self.platform_info = {
            'platform': PLATFORM,
            'is_macos': IS_MACOS,
            'is_apple_silicon': IS_APPLE_SILICON,
            'is_linux': IS_LINUX,
            'is_windows': IS_WINDOWS
        }
        
        # Initialize GPU monitoring based on platform
        self._init_gpu_monitoring()

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring based on the platform"""
        if self.nvidia_available:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_monitoring_available = True
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
                print(f"✅ NVIDIA GPU monitoring initialized: {gpu_name}")
            except Exception as e:
                print(f"❌ Failed to initialize NVIDIA monitoring: {e}")
                self.nvidia_available = False
        
        elif IS_APPLE_SILICON:
            # Apple Silicon Macs use unified memory, monitor system memory as proxy
            self.gpu_monitoring_available = True
            print("✅ Apple Silicon detected - using system memory monitoring")
            
        elif IS_MACOS:
            # Intel Macs
            self.gpu_monitoring_available = True
            print("✅ Intel Mac detected - using system memory monitoring")
            
        else:
            # Other platforms - use basic system memory monitoring
            self.gpu_monitoring_available = True
            print("✅ Using system memory monitoring for this platform")

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

    def warm_up_model(self) -> bool:
        """Pre-warm the model to match interactive performance"""
        print("🔥 Warming up model...")
        
        try:
            # Simple warm-up request using ollama package
            client = ollama.Client(host=self.ollama_url)
            
            response = client.generate(
                model=self.model_name,
                prompt="Hello, please respond with just 'Ready'",
                stream=False,
                options={
                    "num_predict": 5,
                    "temperature": 0.8,
                    "top_k": 40,
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "repeat_penalty": 1.1
                }
            )
            
            if response and 'response' in response:
                print("✅ Model warmed up successfully")
                print(f"    Response: {response['response'].strip()}")
                return True
            else:
                print("⚠️ Warm-up completed but no response received - continuing anyway")
                return True  # Continue anyway
                
        except Exception as e:
            print(f"⚠️ Model warm-up failed: {e} - continuing anyway")
            return True  # Don't fail the entire benchmark for warm-up issues

    def download_model(self) -> bool:
        """Download the Mistral model if not already available"""
        print(f"🔍 Checking if model {self.model_name} is available...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = response.json().get("models", [])
            
            for model in models:
                if self.model_name in model.get("name", ""):
                    print("=" * 50)
                    print(f"✅ MODEL ALREADY EXISTS - NO DOWNLOAD NEEDED")
                    print("=" * 50)
                    print(f"📦 Model: {self.model_name}")
                    model_size_bytes = model.get('size', 0)
                    if model_size_bytes > 0:
                        model_size_gb = model_size_bytes / (1024**3)
                        print(f"💾 Size: {model_size_gb:.1f} GB ({model_size_bytes:,} bytes)")
                    print(f"📅 Modified: {model.get('modified_at', 'Unknown')}")
                    print(f"🚀 Ready to proceed with benchmarking!")
                    print("=" * 50)
                    return True
            
            print(f"📥 Model not found locally. Downloading {self.model_name}...")
            print("⏳ This may take several minutes (downloading ~4.1GB)...")
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
        """Get GPU/memory usage in MB based on platform"""
        if not self.gpu_monitoring_available:
            return 0, 0
        
        if self.nvidia_available and self.gpu_handle:
            # NVIDIA GPU
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                used_mb = mem_info.used // (1024 * 1024)
                total_mb = mem_info.total // (1024 * 1024)
                return used_mb, total_mb
            except:
                return 0, 0
        
        elif IS_APPLE_SILICON:
            # Apple Silicon - monitor system memory (unified memory architecture)
            try:
                # Get memory pressure and usage
                memory = psutil.virtual_memory()
                total_mb = memory.total // (1024 * 1024)
                used_mb = memory.used // (1024 * 1024)
                
                # Try to get more detailed memory info for Apple Silicon
                try:
                    result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        # Parse vm_stat output for more accurate memory usage
                        lines = result.stdout.split('\n')
                        page_size = 4096  # Default page size
                        
                        # Extract page size if available
                        for line in lines:
                            if 'page size of' in line:
                                page_size = int(line.split('page size of')[1].split()[0])
                                break
                        
                        # Calculate used memory more accurately
                        wired_pages = 0
                        active_pages = 0
                        compressed_pages = 0
                        
                        for line in lines:
                            if 'Pages wired down:' in line:
                                wired_pages = int(line.split(':')[1].strip().rstrip('.'))
                            elif 'Pages active:' in line:
                                active_pages = int(line.split(':')[1].strip().rstrip('.'))
                            elif 'Pages occupied by compressor:' in line:
                                compressed_pages = int(line.split(':')[1].strip().rstrip('.'))
                        
                        used_pages = wired_pages + active_pages + compressed_pages
                        used_mb = (used_pages * page_size) // (1024 * 1024)
                        
                except subprocess.TimeoutExpired:
                    pass
                except:
                    pass
                
                return used_mb, total_mb
            except:
                return 0, 0
        
        elif IS_MACOS:
            # Intel Mac - use system memory
            try:
                memory = psutil.virtual_memory()
                total_mb = memory.total // (1024 * 1024)
                used_mb = memory.used // (1024 * 1024)
                return used_mb, total_mb
            except:
                return 0, 0
        
        else:
            # Other platforms - fallback to system memory
            try:
                memory = psutil.virtual_memory()
                total_mb = memory.total // (1024 * 1024)
                used_mb = memory.used // (1024 * 1024)
                return used_mb, total_mb
            except:
                return 0, 0

    def generate_text(self, context_size: int, target_tokens: int = 100, conversation_mode: bool = False, conversation_turns: int = 3) -> Dict:
        """Generate text with specified context size and measure performance
        
        Args:
            context_size: Target context window size
            target_tokens: Target number of tokens to generate (ignored in conversation mode)
            conversation_mode: If True, simulate a multi-turn conversation
            conversation_turns: Number of conversation turns to simulate
        """
        
        # Initialize conversation variables
        conversation_history = []
        conversation_log = []  # Track actual exchanges for display
        total_response_text = ""
        
        # Measure memory before generation
        gpu_mem_before, gpu_total = self.get_gpu_memory_info()
        cpu_mem_before = psutil.Process().memory_info().rss // (1024 * 1024)
        
        # Timing markers
        request_start = time.time()
        prompt_processed_time = None
        first_token_time = None
        tokens_generated = 0
        
        try:
            # Use ollama package for better control and timing
            client = ollama.Client(host=self.ollama_url)
            
            if conversation_mode:
                # Multi-turn conversation mode
                from utils import generate_conversation_starter, generate_followup_question
                initial_prompt = generate_conversation_starter(context_size)
                conversation_history.append({"role": "user", "content": initial_prompt})
                conversation_log.append({"type": "user", "content": initial_prompt})
                
                for turn in range(conversation_turns):
                    # Build conversation context
                    if turn == 0:
                        # First turn uses the initial prompt
                        current_prompt = initial_prompt
                    else:
                        # Subsequent turns build on conversation history
                        conversation_text = "\n".join([
                            f"User: {msg['content']}" if msg['role'] == 'user' 
                            else f"Assistant: {msg['content']}" 
                            for msg in conversation_history
                        ])
                        
                        # Add a follow-up question
                        followup = generate_followup_question(turn - 1)
                        conversation_text += f"\nUser: {followup}"
                        conversation_history.append({"role": "user", "content": followup})
                        conversation_log.append({"type": "user", "content": followup})
                        current_prompt = conversation_text
                    
                    # Generate response for this turn
                    response_stream = client.generate(
                        model=self.model_name,
                        prompt=current_prompt,
                        stream=True,
                        options={
                            "temperature": 0.8,
                            "num_ctx": context_size,
                            "top_k": 40,
                            "top_p": 0.95,
                            "min_p": 0.05,
                            "repeat_penalty": 1.1
                        }
                    )
                    
                    turn_response = ""
                    
                    # Process streaming response for this turn
                    for chunk in response_stream:
                        current_time = time.time()
                        
                        if prompt_processed_time is None:
                            prompt_processed_time = current_time
                        
                        if first_token_time is None:
                            first_token_time = current_time
                        
                        # Count tokens (approximate - each chunk may have multiple tokens)
                        if 'response' in chunk:
                            chunk_text = chunk['response']
                            turn_response += chunk_text
                            total_response_text += chunk_text
                            # Rough token count (words * 1.3 is a reasonable approximation)
                            tokens_generated += len(chunk_text.split()) * 1.3
                    
                    # Add assistant response to conversation history
                    conversation_history.append({"role": "assistant", "content": turn_response.strip()})
                    conversation_log.append({"type": "assistant", "content": turn_response.strip()})
                    
                    print(f"      Turn {turn + 1}: Generated {len(turn_response.split()) * 1.3:.0f} tokens")
            
            else:
                # Single prompt/response mode (original behavior)
                from utils import generate_test_prompt
                prompt = generate_test_prompt(context_size)
                
                response_stream = client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=True,
                    options={
                        "temperature": 0.8,
                        "num_ctx": context_size,
                        "top_k": 40,
                        "top_p": 0.95,
                        "min_p": 0.05,
                        "repeat_penalty": 1.1
                    }
                )
                
                # Process streaming response for single mode
                for chunk in response_stream:
                    current_time = time.time()
                    
                    # Mark when prompt processing is done (first chunk received)
                    if prompt_processed_time is None:
                        prompt_processed_time = current_time
                    
                    if 'response' in chunk:
                        token_chunk = chunk['response']
                        total_response_text += token_chunk
                        
                        # Count tokens (approximate by splitting on whitespace)
                        new_tokens = len(token_chunk.split())
                        tokens_generated += new_tokens
                        
                        # Record first actual token time
                        if first_token_time is None and tokens_generated > 0:
                            first_token_time = current_time
                    
                    # Stop after reaching target tokens naturally
                    if tokens_generated >= target_tokens:
                        break
                    
                    # Check if generation is complete
                    if chunk.get('done', False):
                        break
            
            end_time = time.time()
            
            # Measure memory after generation
            gpu_mem_after, _ = self.get_gpu_memory_info()
            cpu_mem_after = psutil.Process().memory_info().rss // (1024 * 1024)
            
            # Calculate detailed timing metrics
            total_time = end_time - request_start
            model_load_and_prompt_time = (prompt_processed_time - request_start) if prompt_processed_time else 0
            first_token_latency = (first_token_time - request_start) if first_token_time else 0
            prompt_processing_time = (first_token_time - prompt_processed_time) if (first_token_time and prompt_processed_time) else 0
            pure_generation_time = (end_time - first_token_time) if first_token_time else 0
            
            # Calculate tokens per second for pure generation (what you care about)
            final_tokens = len(total_response_text.split())
            tokens_per_second = final_tokens / pure_generation_time if pure_generation_time > 0 else 0
            
            return {
                "context_size": context_size,
                "success": True,
                "total_time": total_time,
                "model_load_and_prompt_time": model_load_and_prompt_time,
                "prompt_processing_time": prompt_processing_time,
                "pure_generation_time": pure_generation_time,
                "first_token_latency": first_token_latency,
                "tokens_generated": final_tokens,
                "tokens_per_second": tokens_per_second,
                "gpu_mem_before": gpu_mem_before,
                "gpu_mem_after": gpu_mem_after,
                "gpu_mem_used": gpu_mem_after - gpu_mem_before,
                "gpu_total": gpu_total,
                "cpu_mem_before": cpu_mem_before,
                "cpu_mem_after": cpu_mem_after,
                "cpu_mem_used": cpu_mem_after - cpu_mem_before,
                "response_length": len(total_response_text),
                "response_text": total_response_text[:200] + "..." if len(total_response_text) > 200 else total_response_text,
                "conversation_mode": conversation_mode,
                "conversation_turns": conversation_turns if conversation_mode else 1,
                "conversation_log": conversation_log if conversation_mode else []
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
                     step_size: int = 2048,
                     iterations: int = 5,
                     conversation_mode: bool = False,
                     conversation_turns: int = 3) -> List[Dict]:
        """Run benchmark across different context sizes with multiple iterations
        
        Args:
            conversation_mode: If True, simulate multi-turn conversations
            conversation_turns: Number of conversation turns to simulate
        """
        
        context_sizes = list(range(start_context, max_context + 1, step_size))
        total_context_tests = len(context_sizes)
        total_iterations = total_context_tests * iterations
        
        mem_type = "unified memory" if self.platform_info['is_apple_silicon'] else ("VRAM" if self.nvidia_available else "memory")
        print(f"🚀 Starting {mem_type} benchmark...")
        print(f"📊 Context size range: {start_context:,} to {max_context:,} (step: {step_size:,})")
        print(f"🔄 Iterations per context: {iterations}")
        print(f"🔢 Total tests to run: {total_context_tests} context sizes × {iterations} iterations = {total_iterations} tests")
        print("=" * 60)
        
        iteration_count = 0
        
        for context_idx, context_size in enumerate(context_sizes, 1):
            print(f"\n📏 Context Size {context_idx}/{total_context_tests}: {context_size:,} tokens")
            print(f"Running {iterations} iterations for statistical accuracy...")
            
            context_results = []
            
            for iteration in range(1, iterations + 1):
                iteration_count += 1
                print(f"\n🔍 Test {iteration_count}/{total_iterations}: Context {context_size:,} - Iteration {iteration}/{iterations}")
                print(f"⏳ Generating text with {context_size:,} token context...")
                
                result = self.generate_text(context_size, conversation_mode=conversation_mode, conversation_turns=conversation_turns)
                result["iteration"] = iteration
                result["context_test_number"] = context_idx
                context_results.append(result)
                self.results.append(result)
                
                if result["success"]:
                    # Show individual test result
                    print(f"    ✅ Iteration {iteration}: {result['tokens_per_second']:.1f} tokens/sec")
                    print(f"    📝 Tokens Generated: {result['tokens_generated']}")
                    mem_type = "Unified Memory" if self.platform_info['is_apple_silicon'] else ("VRAM" if self.nvidia_available else "Memory")
                    print(f"    🖥️  {mem_type}: {result['gpu_mem_after']:,} MB ({result['gpu_mem_after']/result['gpu_total']*100:.1f}%)")
                    print(f"    ⏱️  Times: Gen={result['pure_generation_time']:.3f}s, Prompt={result['prompt_processing_time']:.3f}s")
                else:
                    print(f"    ❌ Iteration {iteration} FAILED: {result.get('error', 'Unknown error')}")
                    mem_type = "unified memory" if self.platform_info['is_apple_silicon'] else ("VRAM" if self.nvidia_available else "memory")
                    print(f"    🛑 Stopping benchmark - likely reached {mem_type} limit")
                    return self.results
            
            # Calculate and show aggregated statistics for this context size
            successful_results = [r for r in context_results if r["success"]]
            
            if successful_results:
                # Calculate statistics
                tokens_per_sec = [r["tokens_per_second"] for r in successful_results]
                prompt_times = [r["prompt_processing_time"] for r in successful_results]
                generation_times = [r["pure_generation_time"] for r in successful_results]
                vram_usage = [r["gpu_mem_after"] for r in successful_results]
                tokens_generated = [r["tokens_generated"] for r in successful_results]
                
                avg_tps = sum(tokens_per_sec) / len(tokens_per_sec)
                min_tps = min(tokens_per_sec)
                max_tps = max(tokens_per_sec)
                avg_prompt_time = sum(prompt_times) / len(prompt_times)
                avg_vram = sum(vram_usage) / len(vram_usage)
                vram_percent = avg_vram / successful_results[0]["gpu_total"] * 100
                avg_tokens = sum(tokens_generated) / len(tokens_generated)
                min_tokens = min(tokens_generated)
                max_tokens = max(tokens_generated)
                
                print(f"\n📊 CONTEXT {context_size:,} SUMMARY ({len(successful_results)}/{iterations} successful):")
                print(f"    ⚡ Performance: {avg_tps:.1f} tokens/sec (avg), range: {min_tps:.1f}-{max_tps:.1f}")
                print(f"    📝 Tokens Generated: {avg_tokens:.0f} (avg), range: {min_tokens}-{max_tokens}")
                mem_type = "Unified Memory" if self.platform_info['is_apple_silicon'] else ("VRAM" if self.nvidia_available else "Memory")
                print(f"    🖥️  {mem_type} Usage: {avg_vram:,.0f} MB ({vram_percent:.1f}%)")
                print(f"    🧠 Prompt Processing: {avg_prompt_time:.3f}s (avg)")
                
                # Show sample conversation from first successful result
                if successful_results:
                    first_result = successful_results[0]
                    has_log = 'conversation_log' in first_result
                    log_content = first_result.get('conversation_log', []) if has_log else []
                    conv_mode = first_result.get('conversation_mode', False)
                    
                    print(f"    🔍 Conv mode: {conv_mode}, Has log: {has_log}, Log entries: {len(log_content)}")
                    
                    if has_log and log_content:
                        print(f"    💬 Sample Conversation:")
                        sample_log = log_content
                        # Show first few exchanges (limit to 4 entries = 2 full exchanges)
                        for i, entry in enumerate(sample_log[:4]):  
                            if entry['type'] == 'user':
                                user_text = entry['content'][:100] + "..." if len(entry['content']) > 100 else entry['content']
                                print(f"       👤 User: {user_text}")
                            elif entry['type'] == 'assistant':
                                assistant_text = entry['content'][:150] + "..." if len(entry['content']) > 150 else entry['content']
                                print(f"       🤖 Assistant: {assistant_text}")
                        if len(sample_log) > 4:
                            print(f"       ... ({len(sample_log) - 4} more exchanges)")
                
                # Show overall progress
                progress = context_idx / total_context_tests * 100
                bar_length = 30
                filled_length = int(bar_length * context_idx // total_context_tests)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                print(f"    📈 Progress: [{bar}] {progress:.1f}% ({context_idx}/{total_context_tests} context sizes)")
                
            else:
                print(f"\n❌ ALL ITERATIONS FAILED for context size {context_size:,}")

                mem_type = "unified memory" if self.platform_info['is_apple_silicon'] else ("VRAM" if self.nvidia_available else "memory")
                print(f"🛑 Stopping benchmark - likely reached {mem_type} limit")

                break
            
            # Small delay between context size tests
            if context_idx < total_context_tests:
                print(f"  ⏸️  Waiting 2 seconds before next context size...")
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
        
        # Group results by context size and calculate averages for multiple iterations
        context_groups = {}
        for result in successful_results:
            context_size = result["context_size"]
            if context_size not in context_groups:
                context_groups[context_size] = []
            context_groups[context_size].append(result)
        
        # Calculate average performance for each context size
        context_averages = []
        for context_size, results in context_groups.items():
            avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
            avg_prompt_time = sum(r.get("prompt_processing_time", 0) for r in results) / len(results)
            avg_vram = sum(r["gpu_mem_after"] for r in results) / len(results)
            
            context_averages.append({
                "context_size": context_size,
                "tokens_per_second": avg_tps,
                "prompt_processing_time": avg_prompt_time,
                "gpu_mem_after": avg_vram,
                "gpu_mem_percent": avg_vram/results[0]["gpu_total"]*100,
                "pure_generation_time": sum(r.get("pure_generation_time", 0) for r in results) / len(results),
                "model_load_time": sum(r.get("model_load_and_prompt_time", 0) for r in results) / len(results),
                "num_iterations": len(results),
                "min_tps": min(r["tokens_per_second"] for r in results),
                "max_tps": max(r["tokens_per_second"] for r in results),
                "gpu_total": results[0]["gpu_total"]
            })
        
        # Find best average performance (ignoring model load time)
        baseline_tps = max([avg["tokens_per_second"] for avg in context_averages])
        
        # Use 5% margin of error for performance comparisons
        performance_threshold = baseline_tps * 0.95  # Within 5% is considered equal
        
        # Find contexts that perform within 5% margin (using averages)
        acceptable_results = [avg for avg in context_averages 
                            if avg["tokens_per_second"] >= performance_threshold]
        
        if not acceptable_results:
            # Fallback to all results if none meet the 5% threshold
            acceptable_results = context_averages
        
        # Choose the largest context size that maintains performance within 5% margin
        optimal_result = max(acceptable_results, key=lambda x: x["context_size"])
        max_vram_result = max(context_averages, key=lambda x: x["context_size"])
        
        # Calculate efficiency metrics (excluding model load time)
        efficiency_scores = []
        for result in acceptable_results:
            # Efficiency = tokens/sec weighted by prompt processing efficiency
            prompt_efficiency = 1.0 / (1.0 + result["prompt_processing_time"]) if result["prompt_processing_time"] > 0 else 1.0
            efficiency_score = result["tokens_per_second"] * prompt_efficiency
            efficiency_scores.append({
                "context_size": result["context_size"],
                "efficiency_score": efficiency_score,
                "tokens_per_second": result["tokens_per_second"],
                "prompt_processing_time": result["prompt_processing_time"]
            })
        
        best_efficiency = max(efficiency_scores, key=lambda x: x["efficiency_score"])
        
        analysis = {
            "max_successful_context": max_vram_result["context_size"],
            "optimal_context_size": optimal_result["context_size"],
            "baseline_performance": baseline_tps,
            "performance_threshold_5pct": performance_threshold,
            "optimal_tokens_per_second": optimal_result["tokens_per_second"],
            "optimal_prompt_processing": optimal_result["prompt_processing_time"],
            "optimal_gpu_memory_mb": optimal_result["gpu_mem_after"],
            "optimal_gpu_memory_percent": optimal_result["gpu_mem_percent"],
            "max_gpu_memory_used": max_vram_result["gpu_mem_after"],
            "total_gpu_memory": successful_results[0]["gpu_total"],
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "efficiency_sweet_spot": best_efficiency["context_size"],
            "acceptable_contexts_within_5pct": len(acceptable_results),
            "recommendations": {
                "recommended_context_size": optimal_result["context_size"],
                "expected_performance": f"{optimal_result['tokens_per_second']:.1f} tokens/sec",
                "expected_prompt_processing": f"{optimal_result['prompt_processing_time']:.3f}s",
                "vram_usage": f"{optimal_result['gpu_mem_percent']:.1f}% ({optimal_result['gpu_mem_after']:,} MB)",
                "performance_margin": f"Within 5% of peak performance ({baseline_tps:.1f} tokens/sec)",
                "efficiency_note": f"Best efficiency sweet spot at {best_efficiency['context_size']:,} context size"
            }
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
        ax2.set_ylabel(f"{mem_type} (MB)")
        mem_type = "Unified Memory" if self.platform_info['is_apple_silicon'] else ("VRAM" if self.nvidia_available else "Memory")
        ax2.set_title(f"{mem_type} Usage vs Context Size")
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
        ax4.set_title(f"{mem_type} Efficiency vs Context Size")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory Benchmark for Ollama Models (VRAM/Unified Memory)')
    parser.add_argument('--conversation', action='store_true', 
                        help='Enable conversation mode for multi-turn benchmarks')
    parser.add_argument('--turns', type=int, default=3,
                        help='Number of conversation turns (default: 3)')
    parser.add_argument('--start-context', type=int, default=2048,
                        help='Starting context size (default: 2048)')
    parser.add_argument('--max-context', type=int, default=32768,
                        help='Maximum context size (default: 32768)')
    parser.add_argument('--step-size', type=int, default=2048,
                        help='Step size for context increments (default: 2048)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations per context size (default: 5)')
    
    args = parser.parse_args()
    
    benchmark = OllamaVRAMBenchmark()
    
    # Start Ollama service
    if not benchmark.start_ollama_service():
        print("Failed to start Ollama service. Please start it manually with 'ollama serve'")
        sys.exit(1)
    
    # Download model
    if not benchmark.download_model():
        print("Failed to download model")
        sys.exit(1)
    
    # Warm up model
    if not benchmark.warm_up_model():
        print("Model warm-up failed, but continuing...")
    
    # Run benchmark
    print("\n" + "="*60)
    print("STARTING MEMORY BENCHMARK")
    print("="*60)
    
    # Display system information
    print(f"🖥️  Platform: {benchmark.platform_info['platform'].title()}")
    if benchmark.platform_info['is_apple_silicon']:
        print("🧠 Architecture: Apple Silicon (Unified Memory)")
    elif benchmark.platform_info['is_macos']:
        print("🧠 Architecture: Intel Mac")
    elif benchmark.nvidia_available:
        try:
            gpu_name = pynvml.nvmlDeviceGetName(benchmark.gpu_handle).decode('utf-8')
            print(f"🎮 GPU: {gpu_name}")
        except:
            print("🎮 GPU: NVIDIA (detected)")
    
    _, total_mem = benchmark.get_gpu_memory_info()
    if total_mem > 0:
        mem_type = "Unified Memory" if benchmark.platform_info['is_apple_silicon'] else ("VRAM" if benchmark.nvidia_available else "System Memory")
        print(f"💾 Total {mem_type}: {total_mem:,} MB")
    
    print("="*60)
    
    try:
        results = benchmark.run_benchmark(
            start_context=args.start_context,
            max_context=args.max_context,
            step_size=args.step_size,
            iterations=args.iterations,
            conversation_mode=args.conversation,
            conversation_turns=args.turns
        )
        
        print("\n" + "="*60)
        print("BENCHMARK ANALYSIS")
        print("="*60)
        
        analysis = benchmark.analyze_results()
        
        if "error" not in analysis:
            print(f"Maximum successful context size: {analysis['max_successful_context']}")
            print(f"Optimal context size (>80% performance): {analysis['optimal_context_size']}")
            print(f"Baseline performance: {analysis['baseline_performance']:.2f} tokens/sec")
            mem_type = "Unified Memory" if benchmark.platform_info['is_apple_silicon'] else ("VRAM" if benchmark.nvidia_available else "Memory")
            print(f"Maximum {mem_type} used: {analysis['max_gpu_memory_used']}/{analysis['total_gpu_memory']} MB")
            print(f"{mem_type} utilization: {(analysis['max_gpu_memory_used']/analysis['total_gpu_memory']*100):.1f}%")
        
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

