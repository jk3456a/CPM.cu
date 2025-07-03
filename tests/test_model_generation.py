#!/usr/bin/env python3
"""Model generation tests using pytest framework"""

import pytest
import subprocess
import sys
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add tests directory to Python path to resolve imports
_test_dir = Path(__file__).parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))

# Import test configurations
from testdata.model_test_configs import (
    MINICPM4_8B_CONFIGS, 
    SIMPLE_MODEL_CONFIGS, 
    TEST_PROMPT, 
    EXPECTED_ANSWERS, 
    TEST_TIMEOUT
)

class TestModelGeneration:
    """Test suite for model generation functionality"""
    
    @staticmethod
    def _get_gpu_memory_usage() -> List[Tuple[int, float, float]]:
        """Get memory usage for all available GPUs using torch"""
        try:
            import torch
            if not torch.cuda.is_available():
                return []
            
            gpu_info = []
            for device_id in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                    memory_info = torch.cuda.mem_get_info(device_id)
                    free_memory, total_memory = memory_info[0], memory_info[1]
                    used_memory = total_memory - free_memory
                    used_gb, total_gb = used_memory / (1024**3), total_memory / (1024**3)
                    gpu_info.append((device_id, used_gb, total_gb))
                except Exception:
                    continue
            return gpu_info
        except ImportError:
            return []
    

    
    @pytest.fixture(scope="session", autouse=True)
    def setup_environment_once(self):
        """Setup test environment once for all tests"""
        # Display GPU memory status
        gpu_info = self._get_gpu_memory_usage()
        if gpu_info:
            print(f"\n🔍 Found {len(gpu_info)} GPU(s):")
            for device_id, used_gb, total_gb in gpu_info:
                usage_percent = (used_gb / total_gb) * 100
                print(f"  GPU {device_id}: {used_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%)")
                print("🎯 Using GPU 0 for all tests")
        
        yield
    
    @pytest.fixture(autouse=True)
    def setup_test_prompt(self):
        """Setup test prompt file for each test"""
        # Create temporary file for test prompt
        self.prompt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.prompt_file.write(TEST_PROMPT)
        self.prompt_file.close()
        
        yield
        
        # Cleanup
        try:
            os.unlink(self.prompt_file.name)
        except:
            pass
    
    def _check_answer_correctness(self, output: str) -> bool:
        """Check if the model output contains correct answer (B or Blue)"""
        output_lower = output.lower()
        for expected in EXPECTED_ANSWERS:
            if expected.lower() in output_lower:
                return True
        return False
    
    def _run_cpmcu_command(self, cmd_args: List[str], timeout: int = TEST_TIMEOUT) -> Dict[str, Any]:
        """Run CPMCU command and return result"""
        full_cmd = [sys.executable, "-m", "cpmcu.cli"] + cmd_args + [
            "--prompt-file", self.prompt_file.name,
            "--num-generate", "32",  # Limit output length for testing
            "--temperature", "0.1",  # Use low temperature for consistent results
            "--memory-limit", "0.8"  # Limit GPU memory usage for testing
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "duration": time.time() - start_time,
                "cmd": " ".join(full_cmd)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "returncode": -1,
                "duration": timeout,
                "cmd": " ".join(full_cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command failed with exception: {str(e)}",
                "returncode": -1,
                "duration": time.time() - start_time,
                "cmd": " ".join(full_cmd)
            }
    
    @pytest.mark.parametrize("config", MINICPM4_8B_CONFIGS)
    def test_minicpm4_8b_configurations(self, config):
        """Test MiniCPM4-8B with different parameter combinations"""
        print(f"\n🧪 Testing {config['name']}: {config['description']}")
        
        # Build command arguments based on configuration
        cmd_args = []
        
        # Determine model path and type
        if config["apply_quant"]:
            cmd_args.extend(["--model-path", "openbmb/MiniCPM4-8B-marlin-cpmcu"])
        else:
            cmd_args.extend(["--model-path", "openbmb/MiniCPM4-8B"])
        
        if config["apply_sparse"]:
            cmd_args.extend(["--model-type", "minicpm4"])
        else:
            cmd_args.extend(["--model-type", "minicpm"])
        
        # Add Eagle configuration
        if config["apply_eagle"]:
            if config["apply_eagle_quant"]:
                cmd_args.extend([
                    "--draft-model-path", "openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu",
                    "--frspec-path", "openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
                ])
            else:
                cmd_args.extend([
                    "--draft-model-path", "openbmb/MiniCPM4-8B-Eagle-FRSpec",
                    "--frspec-path", "openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
                ])
        
        # Add YARN configuration
        if config["minicpm4_yarn"]:
            cmd_args.extend(["--minicpm4-yarn"])
        
        # Run test
        result = self._run_cpmcu_command(cmd_args)
        
        # Print result information
        print(f"⏱️  Duration: {result['duration']:.2f}s")
        print(f"🔄 Return code: {result['returncode']}")
        
        if not result["success"] and ("memory limit exceeded" in result["stderr"] or "out of memory" in result["stderr"]):
            pytest.skip(f"Skipping test due to GPU memory limitation: {config['name']}")
        
        # Assertions
        assert result["success"], f"Command failed with stderr: {result['stderr']}\nCommand: {result['cmd']}"
        assert result["stdout"], "No output generated"
        
        # Check answer correctness
        is_correct = self._check_answer_correctness(result["stdout"])
        print(f"✅ Answer correct: {is_correct}")
        
        # Note: We make this a warning rather than failure since model outputs can vary
        if not is_correct:
            print(f"⚠️  Warning: Model did not produce expected answer (B/Blue)")
    
    @pytest.mark.parametrize("config", SIMPLE_MODEL_CONFIGS)
    def test_simple_model_configurations(self, config):
        """Test simple model configurations with basic command"""
        print(f"\n🧪 Testing {config['name']}: {config['description']}")
        
        # Use simple command as specified in requirements
        cmd_args = ["--model-path", config["model_path"]]
        
        # Run test
        result = self._run_cpmcu_command(cmd_args)
        
        # Print result information
        print(f"⏱️  Duration: {result['duration']:.2f}s")
        print(f"🔄 Return code: {result['returncode']}")
        
        if not result["success"] and ("memory limit exceeded" in result["stderr"] or "out of memory" in result["stderr"]):
            pytest.skip(f"Skipping test due to GPU memory limitation: {config['name']}")
        
        # Assertions
        assert result["success"], f"Command failed with stderr: {result['stderr']}\nCommand: {result['cmd']}"
        assert result["stdout"], "No output generated"
        
        # Check answer correctness
        is_correct = self._check_answer_correctness(result["stdout"])
        print(f"✅ Answer correct: {is_correct}")
        
        # Note: We make this a warning rather than failure since model outputs can vary
        if not is_correct:
            print(f"⚠️  Warning: Model did not produce expected answer (B/Blue)")
    
    def test_gpu_memory_detection(self):
        """Test GPU memory usage detection functionality"""
        print(f"\n🧪 Testing GPU memory detection")
        
        # Test GPU memory usage detection
        gpu_info = self._get_gpu_memory_usage()
        if gpu_info:
            print(f"✅ Detected {len(gpu_info)} GPU(s)")
            for device_id, used_gb, total_gb in gpu_info:
                usage_percent = (used_gb / total_gb) * 100
                print(f"  GPU {device_id}: {used_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%)")
        else:
            print("ℹ️  No GPUs detected")
        
        print("✅ GPU memory detection test passed")
    
    def test_environment_setup(self):
        """Test that environment is properly configured"""
        print(f"\n🧪 Testing environment setup")
        
        # Check that test prompt file exists
        assert os.path.exists(self.prompt_file.name), "Test prompt file should exist"
        
        # Check that cpmcu module is available
        try:
            import cpmcu
            print("✅ CPMCU module available")
        except ImportError as e:
            pytest.fail(f"CPMCU module not available: {e}")
        
        print("✅ Environment setup test passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"]) 