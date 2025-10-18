"""
Unit tests for HardwareDetector class.
"""

import pytest
from unittest.mock import Mock, patch
from models.detector import HardwareDetector, HardwareSpecs


class TestHardwareSpecs:
    """Test HardwareSpecs dataclass."""
    
    def test_hardware_specs_creation(self):
        """Test HardwareSpecs creation with all parameters."""
        specs = HardwareSpecs(
            cpu_cores=8,
            total_memory_gb=16.0,
            available_memory_gb=12.0,
            has_gpu=True,
            gpu_memory_gb=8.0,
            gpu_name="NVIDIA RTX 3060",
            platform="windows",
            architecture="x86_64"
        )
        
        assert specs.cpu_cores == 8
        assert specs.total_memory_gb == 16.0
        assert specs.available_memory_gb == 12.0
        assert specs.has_gpu is True
        assert specs.gpu_memory_gb == 8.0
        assert specs.gpu_name == "NVIDIA RTX 3060"
        assert specs.platform == "windows"
        assert specs.architecture == "x86_64"
    
    def test_hardware_specs_defaults(self):
        """Test HardwareSpecs with default values."""
        specs = HardwareSpecs(
            cpu_cores=4,
            total_memory_gb=8.0,
            available_memory_gb=6.0,
            has_gpu=False
        )
        
        assert specs.gpu_memory_gb is None
        assert specs.gpu_name is None
        assert specs.platform == "unknown"
        assert specs.architecture == "unknown"


class TestHardwareDetector:
    """Test HardwareDetector class."""
    
    @patch('models.detector.psutil.cpu_count')
    @patch('models.detector.psutil.virtual_memory')
    @patch('models.detector.platform.system')
    @patch('models.detector.platform.machine')
    @patch('models.detector.torch.cuda.is_available')
    @patch('models.detector.torch.cuda.get_device_properties')
    @patch('models.detector.torch.cuda.get_device_name')
    def test_detect_hardware_with_gpu(
        self, mock_name, mock_props, mock_cuda, mock_machine, 
        mock_system, mock_memory, mock_cpu
    ):
        """Test hardware detection with GPU."""
        # Setup mocks
        mock_cpu.return_value = 8
        mock_memory.return_value.total = 16 * 1024**3
        mock_memory.return_value.available = 12 * 1024**3
        mock_system.return_value = "Windows"
        mock_machine.return_value = "AMD64"
        mock_cuda.return_value = True
        mock_props.return_value.total_memory = 8 * 1024**3
        mock_name.return_value = "NVIDIA RTX 3060"
        
        detector = HardwareDetector()
        
        assert detector.specs.cpu_cores == 8
        assert detector.specs.total_memory_gb == 16.0
        assert detector.specs.available_memory_gb == 12.0
        assert detector.specs.has_gpu is True
        assert detector.specs.gpu_memory_gb == 8.0
        assert detector.specs.gpu_name == "NVIDIA RTX 3060"
        assert detector.specs.platform == "windows"
        assert detector.specs.architecture == "amd64"
    
    @patch('models.detector.psutil.cpu_count')
    @patch('models.detector.psutil.virtual_memory')
    @patch('models.detector.platform.system')
    @patch('models.detector.platform.machine')
    @patch('models.detector.torch.cuda.is_available')
    def test_detect_hardware_without_gpu(
        self, mock_cuda, mock_machine, mock_system, mock_memory, mock_cpu
    ):
        """Test hardware detection without GPU."""
        # Setup mocks
        mock_cpu.return_value = 4
        mock_memory.return_value.total = 8 * 1024**3
        mock_memory.return_value.available = 6 * 1024**3
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        mock_cuda.return_value = False
        
        detector = HardwareDetector()
        
        assert detector.specs.cpu_cores == 4
        assert detector.specs.total_memory_gb == 8.0
        assert detector.specs.available_memory_gb == 6.0
        assert detector.specs.has_gpu is False
        assert detector.specs.gpu_memory_gb is None
        assert detector.specs.gpu_name is None
        assert detector.specs.platform == "linux"
        assert detector.specs.architecture == "x86_64"
    
    def test_get_compatible_models_high_end_gpu(self, mock_hardware_detector):
        """Test compatible models for high-end GPU."""
        mock_hardware_detector.specs.has_gpu = True
        mock_hardware_detector.specs.gpu_memory_gb = 10.0
        
        models = mock_hardware_detector.get_compatible_models()
        
        # Should include high-end models
        model_names = [m["name"] for m in models]
        assert "Qwen2.5-7B-Instruct" in model_names
        assert "Qwen2.5-14B-Instruct" in model_names or "Qwen2.5-3B-Instruct" in model_names
    
    def test_get_compatible_models_mid_range_gpu(self, mock_hardware_detector):
        """Test compatible models for mid-range GPU."""
        mock_hardware_detector.specs.has_gpu = True
        mock_hardware_detector.specs.gpu_memory_gb = 6.0
        
        models = mock_hardware_detector.get_compatible_models()
        
        # Should include mid-range models
        model_names = [m["name"] for m in models]
        assert "Qwen2.5-3B-Instruct" in model_names
        assert len(models) > 0
    
    def test_get_compatible_models_cpu_only(self, mock_hardware_detector):
        """Test compatible models for CPU-only system."""
        mock_hardware_detector.specs.has_gpu = False
        mock_hardware_detector.specs.available_memory_gb = 8.0
        
        models = mock_hardware_detector.get_compatible_models()
        
        # Should include CPU models
        model_names = [m["name"] for m in models]
        assert "Qwen3-4B-Instruct-CPU" in model_names or "Qwen3-0.6B" in model_names
        assert len(models) > 0
    
    def test_get_compatible_models_mobile(self, mock_hardware_detector):
        """Test compatible models for mobile/edge devices."""
        mock_hardware_detector.specs.has_gpu = False
        mock_hardware_detector.specs.available_memory_gb = 2.0
        
        models = mock_hardware_detector.get_compatible_models()
        
        # Should include mobile-optimized models
        mobile_models = [m for m in models if m.get("mobile_optimized")]
        assert len(mobile_models) > 0
        assert mobile_models[0]["name"] == "Qwen3-0.6B"
    
    def test_estimate_performance_gpu_efficient(self, mock_hardware_detector):
        """Test performance estimation for GPU-efficient model."""
        mock_hardware_detector.specs.has_gpu = True
        mock_hardware_detector.specs.gpu_memory_gb = 8.0
        
        performance = mock_hardware_detector.estimate_performance(6.0)  # 6GB model
        
        assert performance["device"] == "cuda"
        assert performance["memory_efficient"] is True
        assert performance["recommended"] is True
        assert performance["estimated_tokens_per_second"] == 50
    
    def test_estimate_performance_gpu_inefficient(self, mock_hardware_detector):
        """Test performance estimation for GPU-inefficient model."""
        mock_hardware_detector.specs.has_gpu = True
        mock_hardware_detector.specs.gpu_memory_gb = 4.0
        
        performance = mock_hardware_detector.estimate_performance(8.0)  # 8GB model
        
        assert performance["device"] == "cpu"
        assert performance["memory_efficient"] is False
        assert performance["recommended"] is False
        assert performance["estimated_tokens_per_second"] == 5
    
    def test_estimate_performance_cpu_efficient(self, mock_hardware_detector):
        """Test performance estimation for CPU-efficient model."""
        mock_hardware_detector.specs.has_gpu = False
        mock_hardware_detector.specs.available_memory_gb = 8.0
        
        performance = mock_hardware_detector.estimate_performance(5.0)  # 5GB model
        
        assert performance["device"] == "cpu"
        assert performance["memory_efficient"] is True
        assert performance["recommended"] is True
        assert performance["estimated_tokens_per_second"] == 3
    
    def test_estimate_performance_cpu_inefficient(self, mock_hardware_detector):
        """Test performance estimation for CPU-inefficient model."""
        mock_hardware_detector.specs.has_gpu = False
        mock_hardware_detector.specs.available_memory_gb = 4.0
        
        performance = mock_hardware_detector.estimate_performance(6.0)  # 6GB model
        
        assert performance["device"] == "cpu"
        assert performance["memory_efficient"] is False
        assert performance["recommended"] is False
        assert performance["estimated_tokens_per_second"] == 1
    
    def test_get_system_info(self, mock_hardware_detector):
        """Test system information retrieval."""
        system_info = mock_hardware_detector.get_system_info()
        
        assert "hardware" in system_info
        assert "compatible_models" in system_info
        assert "recommendations" in system_info
        
        hardware = system_info["hardware"]
        assert hardware["cpu_cores"] == 8
        assert hardware["total_memory_gb"] == 16.0
        assert hardware["has_gpu"] is True
        
        recommendations = system_info["recommendations"]
        assert "best_model" in recommendations
        assert "mobile_optimized" in recommendations
