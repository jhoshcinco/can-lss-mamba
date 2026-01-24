"""Training utilities and WandB integration for CAN-LSS-Mamba."""

import os
from typing import Optional, Dict, Any


class WandBLogger:
    """Wrapper for Weights & Biases logging."""
    
    def __init__(self, enabled: bool = False, project: str = "can-lss-mamba", 
                 entity: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                 tags: Optional[list] = None, notes: str = ""):
        """
        Initialize WandB logger.
        
        Args:
            enabled: Whether WandB logging is enabled
            project: WandB project name
            entity: WandB entity (username or team)
            config: Configuration dictionary to log
            tags: List of tags for this run
            notes: Notes for this run
        """
        self.enabled = enabled
        self.wandb = None
        self.run = None
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                
                # Initialize WandB
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    config=config or {},
                    tags=tags or [],
                    notes=notes
                )
                print(f"✅ WandB initialized: {wandb.run.url}")
            except ImportError:
                print("⚠️  Warning: wandb not installed. Install with: pip install wandb")
                self.enabled = False
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize WandB: {e}")
                self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.enabled and self.wandb is not None:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:
                print(f"⚠️  Warning: Failed to log to WandB: {e}")
    
    def save(self, path: str, base_path: Optional[str] = None):
        """
        Save a file as a WandB artifact.
        
        Args:
            path: Path to file to save
            base_path: Base path for the artifact
        """
        if self.enabled and self.wandb is not None:
            try:
                self.wandb.save(path, base_path=base_path)
            except Exception as e:
                print(f"⚠️  Warning: Failed to save to WandB: {e}")
    
    def finish(self):
        """Finish the WandB run."""
        if self.enabled and self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception as e:
                print(f"⚠️  Warning: Failed to finish WandB run: {e}")
    
    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch model gradients and parameters.
        
        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all", None)
            log_freq: Logging frequency
        """
        if self.enabled and self.wandb is not None:
            try:
                self.wandb.watch(model, log=log, log_freq=log_freq)
            except Exception as e:
                print(f"⚠️  Warning: Failed to watch model: {e}")


def setup_wandb_from_config(cfg, additional_config: Optional[Dict[str, Any]] = None) -> WandBLogger:
    """
    Setup WandB logger from configuration.
    
    Args:
        cfg: OmegaConf configuration object
        additional_config: Additional config items to log
    
    Returns:
        WandBLogger instance
    """
    # Check if WandB is enabled
    enabled = os.getenv("WANDB_ENABLED", "false").lower() == "true"
    if cfg and hasattr(cfg, "wandb"):
        enabled = cfg.wandb.get("enabled", False)
    
    # Get WandB configuration
    project = os.getenv("WANDB_PROJECT", "can-lss-mamba")
    entity = os.getenv("WANDB_ENTITY", None)
    
    if cfg and hasattr(cfg, "wandb"):
        project = cfg.wandb.get("project", project)
        entity = cfg.wandb.get("entity", entity)
    
    # Prepare config to log
    config_dict = additional_config or {}
    
    # Add key config items
    if cfg:
        if hasattr(cfg, "training"):
            config_dict.update({
                "learning_rate": cfg.training.get("learning_rate", None),
                "batch_size": cfg.training.get("batch_size", None),
                "epochs": cfg.training.get("epochs", None),
                "optimizer": cfg.training.get("optimizer", None),
            })
        if hasattr(cfg, "preprocessing"):
            config_dict.update({
                "window_size": cfg.preprocessing.get("window_size", None),
                "stride": cfg.preprocessing.get("stride", None),
            })
    
    # Get tags
    tags = []
    if cfg and hasattr(cfg, "wandb"):
        tags = cfg.wandb.get("tags", [])
    
    return WandBLogger(
        enabled=enabled,
        project=project,
        entity=entity,
        config=config_dict,
        tags=tags
    )
