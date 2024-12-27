from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseProcessor(ABC):
    """Abstract base class for all transforms in the pipeline."""
    
    def __init__(
        self,
        name: str,
        device: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize transform.
        
        Args:
            name: Unique identifier for this transform
            device: Device to run transform on
            config: Optional configuration dictionary
        """
        self.name = name
        self.device = device
        self.config = config or {}
        
    @abstractmethod
    def process(self, x: Tensor) -> Tensor:
        """Process input tensor.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Processed tensor
        """
        pass
    
    @abstractmethod
    def get_scale_factor(self) -> float:
        """Get the scale factor of this transform.
        
        Returns:
            Scale factor (e.g., 1.0 for same size, 4.0 for 4x upscale)
        """
        pass
    
    def warm_up(self):
        """Optional warm-up routine before processing."""
        pass
    
    def clean_up(self):
        """Optional cleanup routine after processing."""
        pass