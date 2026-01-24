"""Configuration management utilities for CAN-LSS-Mamba."""

import os
from pathlib import Path
from typing import Optional, Any, Dict
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration from YAML file with environment variable overrides.
    
    Priority order (highest to lowest):
    1. Environment variables
    2. Specified config file
    3. Default config file
    
    Args:
        config_path: Path to config file. If None, will check CONFIG_PATH env var,
                    then fall back to configs/default.yaml
    
    Returns:
        OmegaConf DictConfig object with resolved configuration
    """
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Determine which config file to use
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    
    # Convert to Path object for easier manipulation
    config_file = Path(config_path)
    
    # Check if config file exists
    if not config_file.exists():
        # Try relative to project root - look for marker files
        project_root = Path(__file__).parent.parent.parent
        
        # Try to find project root by looking for marker files
        current = Path(__file__).parent
        for _ in range(5):  # Search up to 5 levels
            if (current / "setup.sh").exists() or (current / "requirements.txt").exists():
                project_root = current
                break
            current = current.parent
        
        config_file = project_root / config_path
        
        if not config_file.exists():
            print(f"⚠️  Warning: Config file '{config_path}' not found. Using defaults.")
            return OmegaConf.create({})
    
    # Load config file
    cfg = OmegaConf.load(config_file)
    
    # Resolve interpolations (e.g., ${data.root})
    OmegaConf.resolve(cfg)
    
    return cfg


def get_from_config_or_env(
    cfg: Optional[DictConfig],
    config_key: str,
    env_var: str,
    default: Any = None,
    cast_type: type = str
) -> Any:
    """
    Get a value from config or environment variable with fallback to default.
    
    Priority:
    1. Environment variable
    2. Config file value
    3. Default value
    
    Args:
        cfg: OmegaConf config object
        config_key: Dot-separated key path in config (e.g., "data.root")
        env_var: Environment variable name
        default: Default value if neither config nor env var is set
        cast_type: Type to cast the final value to
    
    Returns:
        The value with appropriate type casting
    """
    # Check environment variable first
    env_value = os.getenv(env_var)
    if env_value is not None:
        try:
            return cast_type(env_value)
        except (ValueError, TypeError):
            print(f"⚠️  Warning: Could not cast env var {env_var}={env_value} to {cast_type}")
    
    # Check config file
    if cfg is not None:
        try:
            keys = config_key.split(".")
            value = cfg
            for key in keys:
                value = value[key]
            if value is not None:
                return cast_type(value)
        except (KeyError, AttributeError):
            pass
    
    # Return default
    return default


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """
    Convert OmegaConf DictConfig to a standard Python dictionary.
    
    Args:
        cfg: OmegaConf config object
    
    Returns:
        Standard Python dict
    """
    return OmegaConf.to_container(cfg, resolve=True)


def print_config(cfg: DictConfig, title: str = "Configuration"):
    """
    Pretty print configuration.
    
    Args:
        cfg: OmegaConf config object
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(OmegaConf.to_yaml(cfg))
    print(f"{'='*60}\n")
