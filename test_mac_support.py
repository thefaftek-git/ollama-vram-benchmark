#!/usr/bin/env python3
"""
Test script to verify Mac support works without NVIDIA GPU requirements
"""

def test_requirements_check():
    """Test that requirements check passes on Mac without NVIDIA"""
    print("🧪 Testing requirements check...")
    
    from utils import check_system_requirements
    requirements = check_system_requirements()
    
    print("Requirements status:")
    for req, status in requirements.items():
        status_str = '✅ PASS' if status else '❌ FAIL'
        print(f"  {req.replace('_', ' ').title()}: {status_str}")
    
    failed = [req for req, status in requirements.items() if not status]
    
    if failed:
        print(f"\n❌ Failed requirements: {failed}")
        return False
    else:
        print(f"\n✅ All requirements passed!")
        return True

def test_benchmark_init():
    """Test that benchmark initializes without NVIDIA GPU"""
    print("\n🧪 Testing benchmark initialization...")
    
    try:
        from benchmark import OllamaVRAMBenchmark
        benchmark = OllamaVRAMBenchmark()
        
        print(f"✅ Benchmark initialized successfully!")
        print(f"  Platform: {benchmark.platform_info['platform']}")
        print(f"  Apple Silicon: {benchmark.platform_info['is_apple_silicon']}")
        print(f"  macOS: {benchmark.platform_info['is_macos']}")
        print(f"  GPU monitoring available: {benchmark.gpu_monitoring_available}")
        print(f"  NVIDIA available: {benchmark.nvidia_available}")
        
        # Test memory reading
        used, total = benchmark.get_gpu_memory_info()
        if total > 0:
            print(f"  Memory monitoring: {used:,} MB used of {total:,} MB total")
            return True
        else:
            print(f"  ❌ Memory monitoring returned 0 total memory")
            return False
            
    except Exception as e:
        print(f"❌ Benchmark initialization failed: {e}")
        return False

def test_run_benchmark_script():
    """Test that run_benchmark.py would pass requirements"""
    print("\n🧪 Testing run_benchmark.py requirements...")
    
    try:
        from run_benchmark import parse_arguments
        from utils import check_system_requirements
        
        requirements = check_system_requirements()
        failed_requirements = [req for req, status in requirements.items() if not status]
        
        if failed_requirements:
            print(f"❌ run_benchmark.py would fail with: {failed_requirements}")
            return False
        else:
            print(f"✅ run_benchmark.py would pass requirements check!")
            return True
            
    except Exception as e:
        print(f"❌ run_benchmark.py test failed: {e}")
        return False

if __name__ == "__main__":
    print("🍎 Mac Support Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_requirements_check():
        tests_passed += 1
    
    if test_benchmark_init():
        tests_passed += 1
        
    if test_run_benchmark_script():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Mac support is working correctly.")
        print("\nYou can now run the benchmark on Mac with:")
        print("  python run_benchmark.py --quick")
    else:
        print("❌ Some tests failed. Mac support may have issues.")
        
    print("\n💡 Note: Ollama service needs to be running for actual benchmarks")
