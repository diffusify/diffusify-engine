import os
import torch
import torch.cuda.amp as amp
import torch._dynamo
from torch import Tensor
from safetensors.torch import load_file
from spandrel import ModelLoader
from tqdm import tqdm
from threading import Event
from typing import Optional, Dict, Any

from .base import BaseProcessor
from .utils.helpers import tiled_scale

class SpandrelProcessor(BaseProcessor):
    """Spandrel-loaded model processor with built-in tiling support."""
    
    def __init__(
        self,
        name: str,
        model_path: str,
        device: str,
        config: Optional[Dict[str, Any]] = None,
        tile_size: int = 512,
        min_tile_size: int = 128,
        overlap: int = 32,
        batch_size: int = 1
    ):
        """Initialize model processor.
        
        Args:
            name: Unique identifier for this processor
            model_path: Path to model weights file
            device: Device to run model on
            config: Optional configuration dictionary
            tile_size: Initial tile size for processing
            min_tile_size: Minimum tile size before failing
            overlap: Overlap between tiles
            batch_size: Batch size for processing
        """
        super().__init__(name, device, config)

        # Add FP16 configuration
        self.use_amp = config.get('use_amp', True)
        self.compile_model = config.get('compile_model', False)

        # Model configuration
        self.model_path = model_path
        self.model = None
        self.scale_factor = None

        # Tiling parameters
        self.tile_size = config.get('tile_size', 512)
        self.min_tile_size = config.get('min_tile_size', 128)
        self.overlap = overlap
        self.batch_size = batch_size

        # Memory tracking
        self.peak_memory = 0

        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load model from weights file and determine scale factor."""
        try:
            print(f"Loading model {self.name} from {self.model_path}")

            # Load state dict based on file extension
            if self.model_path.endswith('.safetensors'):
                state_dict = load_file(self.model_path)
            else:  # Handle .pth and other formats
                state_dict = torch.load(self.model_path, map_location=self.device)
            
            
            self.model = ModelLoader(device=self.device).load_from_state_dict(state_dict)

            # Optional model compilation with progress feedback
            if self.compile_model and hasattr(torch, 'compile'):
                print(f"Compiling model {self.name} (this may take a few minutes)...")
                try:
                    self.model = torch.compile(
                        self.model,
                        fullgraph=True,
                        dynamic=True,
                        mode='reduce-overhead'  # or 'max-autotune' for more optimization
                    )
                    print(f"✓ Model compilation complete")
                except Exception as e:
                    print(f"Warning: Model compilation failed, falling back to non-compiled version: {str(e)}")
            
            # Determine scale factor
            if hasattr(self.model, 'scale'):
                self.scale_factor = float(self.model.scale)
            else:
                # Use configuration
                self.scale_factor = 4.0
                raise RuntimeError(f"Model has an unknown scale factor, implement better model configuration")
                
            print(f"✓ Model {self.name} loaded successfully (scale factor: {self.scale_factor}x, "
                f"compiled: {self.compile_model}, "
                f"AMP enabled: {self.use_amp})")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.name}: {str(e)}")

    def process(
        self,
        x: Tensor,
        pbar: Optional[tqdm] = None,
        interrupt_flag: Optional[Event] = None
    ) -> Tensor:
        """Process input tensor with automatic tiling.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            pbar: Optional progress bar
            interrupt_flag: Optional interrupt flag
            
        Returns:
            Processed tensor
        
        Raises:
            RuntimeError: If processing fails even with minimum tile size
        """
        if self.model is None:
            raise RuntimeError(f"Model {self.name} not loaded")
        try:
            # Move input to target device
            x = x.to(self.device)
            
            # Process using tiled_scale
            # with amp.autocast('cuda', enabled=self.use_amp):
            output = tiled_scale(
                image=x,
                function=self.model,
                tile_x=self.tile_size,
                tile_y=self.tile_size,
                overlap=self.overlap,
                upscale_amount=self.scale_factor,
                output_device=self.device,
                pbar=pbar,
                interrupt_flag=interrupt_flag
            )

            # Ensure output is float32
            if output.dtype != torch.float32:
                output = output.float()
            
            # Update memory statistics
            self._update_memory_stats()
            
            return output
            
        except torch.cuda.OutOfMemoryError:
            # Attempt recovery with smaller tile size
            current_tile = self.tile_size
            while current_tile >= self.min_tile_size:
                try:
                    current_tile //= 2
                    print(f"\nOOM error in {self.name}, reducing tile size to {current_tile}")
                    
                    # Clear cache and retry
                    torch.cuda.empty_cache()
                    
                    output = tiled_scale(
                        image=x,
                        function=self.model,
                        tile_x=current_tile,
                        tile_y=current_tile,
                        overlap=self.overlap,
                        upscale_amount=self.scale_factor,
                        output_device=self.device,
                        pbar=pbar,
                        interrupt_flag=interrupt_flag
                    )
                    
                    self._update_memory_stats()
                    return output

                except torch.cuda.OutOfMemoryError:
                    if current_tile <= self.min_tile_size:
                        raise RuntimeError(
                            f"Failed to process with {self.name} even at minimum tile size"
                        )
                    continue

        except Exception as e:
            raise RuntimeError(f"Error processing with {self.name}: {str(e)}")
    
    def get_scale_factor(self) -> float:
        """Get model's scale factor.
        
        Returns:
            Scale factor from model
        """
        if hasattr(self.model, 'scale'):
            return float(self.model.scale)
        return 1.0
    
    def warm_up(self):
        """Warm up model with small test input."""
        try:
            print(f"Warming up {self.name}...")
            with torch.inference_mode():
                test_input = torch.randn(
                    1, 3, self.tile_size, self.tile_size,
                    device=self.device
                )
                _ = self.process(test_input)
            print(f"✓ {self.name} warm-up complete")
        except Exception as e:
            print(f"Warning: Failed to warm up {self.name}: {str(e)}")
    
    def clean_up(self):
        """Clean up model resources."""
        if self.model is not None:
            try:
                # Delete model and clear cache
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"✓ {self.name} cleaned up successfully")
            except Exception as e:
                print(f"Warning: Error cleaning up {self.name}: {str(e)}")
    
    def _update_memory_stats(self):
        """Update peak memory usage statistics."""
        if torch.cuda.is_available():
            current = torch.cuda.max_memory_allocated(self.device)
            self.peak_memory = max(self.peak_memory, current)

    def __str__(self) -> str:
        """Get string representation of processor.
        
        Returns:
            Processor description string
        """
        return (
            f"SpandrelProcessor(name={self.name}, "
            f"device={self.device}, "
            f"scale={self.scale_factor}x, "
            f"tile={self.tile_size})"
        )
