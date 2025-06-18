
#!/usr/bin/env python3
"""
Main script to run the VRAM benchmark with command line options
"""

import argparse
import sys
import os
from datetime import datetime

from benchmark import OllamaVRAMBenchmark
from utils import check_system_requirements, print_system_info
from config import BENCHMARK_CONFIG, MODEL_CONFIG, HARDWARE_CONFIG


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama VRAM usage for Mistral-7B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py                          # Run with default settings (5 iterations per context)
  python run_benchmark.py --start 4096 --max 16384 --step 4096  # Custom range
  python run_benchmark.py --quick                  # Quick test mode
  python run_benchmark.py --iterations 10          # More iterations for higher accuracy
  python run_benchmark.py --iterations 1           # Single iteration for speed
  python run_benchmark.py --info                   # Show system info only
        """
    )
    
    parser.add_argument(
        "--start", 
        type=int, 
        default=BENCHMARK_CONFIG["start_context"],
        help=f"Starting context size (default: {BENCHMARK_CONFIG['start_context']})"
    )
    
    parser.add_argument(
        "--max", 
        type=int, 
        default=BENCHMARK_CONFIG["max_context"],
        help=f"Maximum context size (default: {BENCHMARK_CONFIG['max_context']})"
    )
    
    parser.add_argument(
        "--step", 
        type=int, 
        default=BENCHMARK_CONFIG["step_size"],
        help=f"Step size between tests (default: {BENCHMARK_CONFIG['step_size']})"
    )
    
    parser.add_argument(
        "--tokens", 
        type=int, 
        default=BENCHMARK_CONFIG["num_tokens_generate"],
        help=f"Number of tokens to generate per test (default: {BENCHMARK_CONFIG['num_tokens_generate']})"
    )
    
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=5,
        help="Number of iterations to run per context size for statistical accuracy (default: 5)"
    )
    
    parser.add_argument(
        "--conversation", 
        action="store_true",
        help="Use conversation mode instead of single prompt/response"
    )
    
    parser.add_argument(
        "--turns", 
        type=int, 
        default=3,
        help="Number of conversation turns in conversation mode (default: 3)"
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick test with smaller range (2048-8192, step 2048)"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Show system information and exit"
    )
    
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Skip generating plots"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Output filename prefix (default: vram_benchmark_TIMESTAMP)"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        default=MODEL_CONFIG["name"],
        help=f"Model name to use (default: {MODEL_CONFIG['name']})"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Show system info and exit if requested
    if args.info:
        print_system_info()
        return
    
    # Check system requirements
    print("Checking system requirements...")
    requirements = check_system_requirements()
    
    failed_requirements = [req for req, status in requirements.items() if not status]
    if failed_requirements:
        print("❌ Missing requirements:")
        for req in failed_requirements:
            print(f"  - {req.replace('_', ' ').title()}")
        
        if "ollama_installed" in failed_requirements:
            print("\nTo install Ollama:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
        
        if "python_packages" in failed_requirements:
            print("\nTo install Python packages:")
            print("  pip install -r requirements.txt")
        
        print("\nPlease install missing requirements and try again.")
        sys.exit(1)
    
    print("✅ All requirements met!")
    
    # Adjust settings for quick mode
    if args.quick:
        args.start = 2048
        args.max = 8192
        args.step = 2048
        print("🚀 Quick mode enabled")
    
    # Show configuration
    print("\nBenchmark Configuration:")
    print("-" * 40)
    print(f"Model: {args.model}")
    print(f"Context range: {args.start} to {args.max} (step: {args.step})")
    print(f"Tokens per test: {args.tokens}")
    print(f"Iterations per context: {args.iterations}")
    if args.conversation:
        print(f"Mode: Conversation ({args.turns} turns per test)")
    else:
        print(f"Mode: Single prompt/response")
    print(f"Expected hardware: {HARDWARE_CONFIG['gpu_name']} ({HARDWARE_CONFIG['vram_gb']}GB)")
    print()
    
    # Initialize benchmark
    benchmark = OllamaVRAMBenchmark()
    benchmark.model_name = args.model
    
    try:
        # Start Ollama service
        print("Starting Ollama service...")
        if not benchmark.start_ollama_service():
            print("❌ Failed to start Ollama service")
            print("Please start it manually with: ollama serve")
            sys.exit(1)
        
        # Download model
        print(f"Ensuring model {args.model} is available...")
        if not benchmark.download_model():
            print("❌ Failed to download model")
            sys.exit(1)
        
        # Show initial system state
        print_system_info()
        
        # Run benchmark
        print("\n" + "="*60)
        print("🚀 STARTING VRAM BENCHMARK")
        print("="*60)
        print("This may take several minutes depending on your context range...")
        print("Press Ctrl+C to stop early (results will still be saved)")
        print()
        
        results = benchmark.run_benchmark(
            start_context=args.start,
            max_context=args.max,
            step_size=args.step,
            iterations=args.iterations,
            conversation_mode=args.conversation,
            conversation_turns=args.turns
        )
        
        if not results:
            print("❌ No results obtained")
            sys.exit(1)
        
        # Analyze results
        print("\n" + "="*60)
        print("📊 BENCHMARK ANALYSIS")
        print("="*60)
        
        analysis = benchmark.analyze_results()
        
        if "error" in analysis:
            print(f"❌ Analysis error: {analysis['error']}")
        else:
            print(f"✅ Maximum successful context size: {analysis['max_successful_context']:,}")
            print(f"🎯 Optimal context size (>80% perf): {analysis['optimal_context_size']:,}")
            print(f"⚡ Baseline performance: {analysis['baseline_performance']:.1f} tokens/sec")
            
            if analysis['total_gpu_memory'] > 0:
                vram_usage = analysis['max_gpu_memory_used'] / analysis['total_gpu_memory'] * 100
                print(f"🖥️  Maximum VRAM usage: {analysis['max_gpu_memory_used']:,}/{analysis['total_gpu_memory']:,} MB ({vram_usage:.1f}%)")
            
            # Compare to expected performance
            expected_tps = MODEL_CONFIG["baseline_tps"]
            actual_tps = analysis['baseline_performance']
            perf_ratio = actual_tps / expected_tps * 100
            
            print(f"📈 Performance vs LMStudio: {perf_ratio:.1f}% ({actual_tps:.1f} vs {expected_tps} tokens/sec)")
            
            # Recommendations
            print("\n💡 Recommendations:")
            if analysis['optimal_context_size'] >= 16384:
                print("  • Your GPU can handle large context sizes efficiently")
            elif analysis['optimal_context_size'] >= 8192:
                print("  • Good performance up to medium context sizes")
            else:
                print("  • Consider smaller context sizes for optimal performance")
            
            if vram_usage > 90:
                print("  • Near VRAM limit - monitor for stability")
            elif vram_usage > 70:
                print("  • Good VRAM utilization")
            else:
                print("  • Room for larger models or higher context sizes")
        
        # Save results
        output_prefix = args.output
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"vram_benchmark_{timestamp}"
        
        benchmark.save_results(output_prefix)
        
        if not args.no_plots:
            benchmark.plot_results(output_prefix)
        
        print(f"\n💾 Results saved with prefix: {output_prefix}")
        print("\nBenchmark completed successfully! 🎉")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        if benchmark.results:
            print("Saving partial results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"vram_benchmark_partial_{timestamp}"
            benchmark.save_results(output_prefix)
            if not args.no_plots:
                benchmark.plot_results(output_prefix)
            print(f"Partial results saved with prefix: {output_prefix}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

