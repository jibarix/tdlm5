"""
Simplified Core utilities for TDLM.

Focused on essential functionality:
- Configuration loading and management
- Basic logging setup  
- Random seed management
- Simple utility functions

Removed: ConfigRegistry, complex environment tracking, hardware validation,
model variants, and other over-engineering.
"""

import os
import sys
import logging
import random
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np
import torch
import yaml


class DictToObj:
    """Convert nested dictionaries to objects for dot notation access."""
    
    def __init__(self, dictionary: Dict[str, Any]):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)


class TDLMConfig:
    """
    Simple configuration container with basic validation.
    
    Provides structured access to configuration with minimal validation
    and convenient attribute access.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary with basic validation."""
        self._raw_config = config_dict.copy()
        self._validate_basic_config()
        
        # Convert nested dicts to objects for dot notation access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)
    
    def _validate_basic_config(self):
        """Basic validation of critical configuration parameters."""
        config = self._raw_config
        
        # Validate model configuration
        if "model" in config:
            model_config = config["model"]
            
            # Check hidden_size is divisible by num_heads
            hidden_size = model_config.get("hidden_size", 384)
            num_heads = model_config.get("num_heads", 6)
            if hidden_size % num_heads != 0:
                raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
            
            # Validate training mode
            training_mode = model_config.get("training_mode", "diffusion")
            if training_mode not in ["diffusion", "autoregressive"]:
                raise ValueError(f"training_mode must be 'diffusion' or 'autoregressive', got: {training_mode}")
        
        # Validate training configuration
        if "training" in config:
            training_config = config["training"]
            
            # Check batch size configuration
            batch_size = training_config.get("batch_size", 32)
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary representation."""
        return self._raw_config.copy()
    
    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, indent=2)
        
        logging.info(f"Configuration saved to: {path}")


def load_config(config_path: Union[str, Path]) -> TDLMConfig:
    """
    Load and validate YAML configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated TDLMConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
    
    if not isinstance(config_dict, dict):
        raise ValueError("Configuration file must contain a dictionary at root level")
    
    logging.info(f"Loaded configuration from: {config_path}")
    return TDLMConfig(config_dict)


def setup_logging(
    config: TDLMConfig,
    experiment_id: Optional[str] = None,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup basic logging system.
    
    Args:
        config: TDLM configuration
        experiment_id: Unique experiment identifier
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if log_dir is None:
        log_dir = Path("logs") / experiment_id
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('tdlm')
    logger.setLevel(level)

    # Log basic system information
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Logging to: {log_dir}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory: {memory_gb:.1f}GB")
    else:
        logger.warning("CUDA not available - using CPU")

    return logger


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUBLAS deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logging.info(f"Set random seed to: {seed}")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes (K, M, B)."""
    if abs(num) >= 1e9:
        return f"{num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def get_device_info() -> Dict[str, Any]:
    """Get basic device information."""
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_available': True,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
        })
    else:
        info['cuda_available'] = False
    
    return info


def create_experiment_dir(base_dir: Union[str, Path], experiment_id: str) -> Path:
    """Create directory structure for experiment."""
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_id
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'samples', 'configs', 'results']
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def generate_experiment_id() -> str:
    """Generate simple experiment ID with timestamp."""
    return f"tdlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# Wandb Integration Utilities

def get_git_commit() -> str:
    """
    Get current git commit hash for experiment tracking.
    
    Returns:
        Git commit hash or 'unknown' if not available
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def get_git_branch() -> str:
    """
    Get current git branch for experiment tracking.
    
    Returns:
        Git branch name or 'unknown' if not available
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed GPU information for wandb logging.
    
    Returns:
        Dictionary with GPU details or empty dict if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return {'cuda_available': False}
    
    try:
        gpu_info = {
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
            'gpu_memory_cached_gb': torch.cuda.memory_reserved(0) / 1e9
        }
        return gpu_info
    except Exception as e:
        logging.warning(f"Failed to get GPU info: {e}")
        return {'cuda_available': True, 'error': str(e)}


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for wandb logging.
    
    Returns:
        Dictionary with system details
    """
    return {
        'platform': platform.platform(),
        'platform_system': platform.system(),
        'platform_release': platform.release(),
        'platform_machine': platform.machine(),
        'platform_processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'hostname': platform.node()
    }


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of key packages for reproducibility.
    
    Returns:
        Dictionary with package versions
    """
    versions = {
        'torch': torch.__version__,
        'python': platform.python_version()
    }
    
    # Try to get other package versions
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        pass
    
    try:
        import transformers
        versions['transformers'] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import datasets
        versions['datasets'] = datasets.__version__
    except ImportError:
        pass
    
    try:
        import wandb
        versions['wandb'] = wandb.__version__
    except ImportError:
        pass
    
    return versions


def get_environment_info() -> Dict[str, Any]:
    """
    Get comprehensive environment information for wandb experiment tracking.
    
    This includes git info, system details, GPU info, and package versions.
    Essential for reproducible research and experiment tracking.
    
    Returns:
        Dictionary with all environment details
    """
    env_info = {
        'timestamp': datetime.now().isoformat(),
        
        # Git information
        'git_commit': get_git_commit(),
        'git_branch': get_git_branch(),
        
        # System information
        **get_system_info(),
        
        # GPU information
        **get_gpu_info(),
        
        # Package versions
        'package_versions': get_package_versions(),
        
        # Environment variables (selected)
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    
    return env_info


def save_environment_info(save_dir: Union[str, Path], experiment_id: str) -> Path:
    """
    Save environment information to file for experiment records.
    
    Args:
        save_dir: Directory to save environment info
        experiment_id: Unique experiment identifier
        
    Returns:
        Path to saved environment file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    env_info = get_environment_info()
    env_info['experiment_id'] = experiment_id
    
    env_file = save_dir / 'environment_info.json'
    
    import json
    with open(env_file, 'w') as f:
        json.dump(env_info, f, indent=2, default=str)
    
    logging.info(f"Environment info saved to: {env_file}")
    return env_file


# Export main components
__all__ = [
    'TDLMConfig', 
    'DictToObj',
    'load_config',
    'setup_logging',
    'set_random_seeds',
    'count_parameters',
    'format_number',
    'get_device_info',
    'create_experiment_dir',
    'generate_experiment_id',
    # Wandb integration utilities
    'get_git_commit',
    'get_git_branch', 
    'get_gpu_info',
    'get_system_info',
    'get_package_versions',
    'get_environment_info',
    'save_environment_info'
]