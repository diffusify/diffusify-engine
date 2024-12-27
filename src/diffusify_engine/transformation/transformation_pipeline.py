import torch

from torch import Tensor
from typing import List, Optional
from threading import Event
from tqdm import tqdm

from .processors.base import BaseProcessor

class TransformationPipeline:
    """Manages a sequence of processor transforms for frame processing."""
    
    def __init__(
        self,
        processors: List[BaseProcessor],
        output_device: str = "cpu"
    ):
        """Initialize processing pipeline.
        
        Args:
            processors: List of BaseProcessor instances to apply in sequence
            output_device: Device to store final output
        """
        self.processors = processors
        self.output_device = output_device
        
        # Validate processors
        if not processors:
            raise ValueError("Pipeline must contain at least one processor")
        
        # Ensure all processors are on the same device as output_device
        if any(p.device != output_device for p in processors):
            raise ValueError(
                f"All processors must be on same device as output_device ({output_device}). "
                f"Found processors on: {[p.device for p in processors]}"
            )
            
        # Calculate total scale factor across all processors
        self.total_scale = 1.0
        for processor in processors:
            self.total_scale *= processor.get_scale_factor()
            
        # Store processor names for logging
        self.processor_names = [p.name for p in processors]
        
    def process_frame(
        self,
        frame: Tensor,
        pbar: Optional[tqdm] = None,
        interrupt_flag: Optional[Event] = None
    ) -> Tensor:
        """Process a single frame through all processors in sequence.
        
        Args:
            frame: Input frame tensor [B, C, H, W]
            pbar: Optional progress bar
            interrupt_flag: Optional interrupt flag
            
        Returns:
            Processed frame tensor
        """
        if frame.device != self.output_device:
            frame = frame.to(self.output_device)

        try:
            # Process through each processor in sequence
            for i, processor in enumerate(self.processors):
                if interrupt_flag and interrupt_flag.is_set():
                    break
                try:
                    # Update progress bar description if available
                    if pbar:
                        pbar.reset()
                        pbar.set_description(
                            f"Processing with (layer{i}) {processor.name}"
                        )
                    
                    # Apply processor transform
                    frame = processor.process(frame, pbar, interrupt_flag)
                        
                except Exception as e:
                    raise RuntimeError(
                        f"Error in processor {processor.name}: {str(e)}"
                    )
                    
            # Ensure output is on specified device
            return frame.to(self.output_device)
            
        except Exception as e:
            raise RuntimeError(f"Pipeline processing error: {str(e)}")
            
    def warm_up(self, sample_input: Optional[Tensor] = None):
        """Warm up all processors in the pipeline."""
        print(f"Warming up pipeline on {self.device}...")
        try:
            if sample_input is None:
                # Create dummy input on correct device
                sample_input = torch.randn(
                    1, 3, 64, 64,
                    device=self.device
                )
            else:
                sample_input = sample_input.to(self.device)
                
            # Warm up each processor in sequence
            with torch.inference_mode():
                x = sample_input
                for processor in self.processors:
                    try:
                        processor.warm_up()
                        x = processor.process(x)
                    except Exception as e:
                        print(f"Warning: Failed to warm up {processor.name}: {e}")
                        
            print(f"✓ Pipeline warm-up complete on {self.device}")
            
        except Exception as e:
            print(f"Warning: Pipeline warm-up failed on {self.device}: {e}")
    
    def clean_up(self):
        """Clean up all processors in the pipeline."""
        print(f"Cleaning up pipeline on {self.device}...")
        try:
            # Clean up processors
            for processor in self.processors:
                processor.clean_up()
                
            # Clear CUDA cache for this device
            if torch.cuda.is_available() and 'cuda' in self.device:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    
            print(f"✓ Pipeline cleanup complete")
        except Exception as e:
            print(f"Warning: Pipeline cleanup failed: {str(e)}")

    @property
    def device(self) -> str:
        """Get pipeline's primary device."""
        return self.output_device

    @property
    def device_mapping(self) -> List[str]:
        """Get list of devices used by processors in sequence.
        
        Returns:
            List of device strings for each processor
        """
        return [p.device for p in self.processors]
    
    def __str__(self) -> str:
        """Get string representation of pipeline.
        
        Returns:
            Pipeline description string
        """
        stages = " -> ".join(self.processor_names)
        return f"ProcessingPipeline({stages})"
    
    def __len__(self) -> int:
        """Get number of processors in pipeline.
        
        Returns:
            Number of processors
        """
        return len(self.processors)
