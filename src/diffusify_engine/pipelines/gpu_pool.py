import os
import torch
from dataclasses import dataclass
from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional

from PIL import Image
from torchvision import transforms

from .transformation_pipeline import TransformationPipeline

@dataclass
class WorkerResult:
    """Results from a GPU worker's processing session."""
    device: str
    processed_frames: List[str]
    failed_frames: List[str]
    peak_memory: float  # Peak memory usage in MB
    avg_frame_time: float  # Average time per frame in seconds
    error: Optional[Exception] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = len(self.processed_frames) + len(self.failed_frames)
        return (len(self.processed_frames) / total * 100) if total > 0 else 0.0

class GPUWorker:
    def __init__(
        self,
        device: str,
        pipeline: TransformationPipeline,
        pool: 'GPUWorkerPool', 
        batch_size: int = 1
    ):
        """Initialize GPU worker.
        
        Args:
            device: GPU device identifier
            pipeline: Pre-configured processing pipeline
            batch_size: Number of frames to process at once
        """
        self.device = device
        self.pool = pool 
        self.pipeline = pipeline
        self.batch_size = batch_size
        
        # Performance tracking
        self.peak_memory = 0
        self.total_time = 0
        self.frames_processed = 0

    def process_frames(
        self,
        frames: List[str],
        input_dir: str,
        output_dir: str,
        interrupt_flag: Optional[Event] = None
    ) -> WorkerResult:
        """Process a list of frames using the pipeline.
        
        Args:
            frames: List of frame filenames
            input_dir: Input directory path
            output_dir: Output directory path
            pbar: Progress bar
            interrupt_flag: Optional interrupt flag
            
        Returns:
            WorkerResult containing processing statistics
        """
        processed_frames = []
        failed_frames = []
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            # Set active device
            torch.cuda.set_device(self.device)
            
            # Warm up pipeline
            # self._warm_up()

            # Initialize batch progress
            self.pool.main_pbar.refresh()
            
            # Process frames in batches
            for i in range(0, len(frames), self.batch_size):
                if interrupt_flag and interrupt_flag.is_set():
                    break

                batch = frames[i:i + self.batch_size]
                batch_tensors = []
                
                # Load batch
                for frame in batch:
                    try:
                        input_path = os.path.join(input_dir, frame)
                        img = Image.open(input_path).convert('RGB')
                        tensor = transforms.ToTensor()(img).unsqueeze(0)
                        batch_tensors.append((frame, tensor))
                    except Exception as e:
                        print(f"\nError loading frame {frame}: {e}")
                        failed_frames.append(frame)
                        continue
                
                # Process batch
                for frame_name, tensor in batch_tensors:
                    if interrupt_flag and interrupt_flag.is_set():
                        break
                    try:
                        # Time the processing
                        start_time.record()
                        
                        # Process through pipeline
                        output = self.pipeline.process_frame(
                            tensor.to(self.device),
                            pbar=self.pool.gpu_pbars[self.device],
                            interrupt_flag=interrupt_flag
                        )
                        
                        end_time.record()
                        torch.cuda.synchronize(device=self.device)
                        
                        # Update timing stats
                        frame_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                        self.total_time += frame_time
                        self.frames_processed += 1
                        
                        # Save output
                        output_path = os.path.join(output_dir, frame_name)
                        output_img = transforms.ToPILImage()(output.squeeze().cpu())
                        output_img.save(output_path, 'PNG')
                        
                        processed_frames.append(frame_name)
                        self._update_memory_stats()
                        
                        # Update progress
                        self.pool.main_pbar.update(1)
                        # self.pool.main_pbar.refresh()  # Force refresh?
                        
                    except Exception as e:
                        print(f"\nError processing frame {frame_name} on {self.device}: {e}")
                        failed_frames.append(frame_name)
                        continue
                        
         
            # Calculate average frame time
            avg_frame_time = (
                self.total_time / self.frames_processed 
                if self.frames_processed > 0 else 0
            )
            
            return WorkerResult(
                device=self.device,
                processed_frames=processed_frames,
                failed_frames=failed_frames,
                peak_memory=self.peak_memory / (1024 * 1024),  # Convert to MB
                avg_frame_time=avg_frame_time
            )
            
        except Exception as e:
            return WorkerResult(
                device=self.device,
                processed_frames=processed_frames,
                failed_frames=failed_frames + [f for f in frames if f not in processed_frames],
                peak_memory=self.peak_memory / (1024 * 1024),
                avg_frame_time=0,
                error=e
            )

    def _update_memory_stats(self):
        """Update peak memory usage statistics."""
        if torch.cuda.is_available():
            current = torch.cuda.max_memory_allocated(self.device)
            self.peak_memory = max(self.peak_memory, current)
            
    def _warm_up(self):
        """Warm up the pipeline with a dummy input."""
        try:
            print(f"Warming up worker on {self.device}...")
            self.pipeline.warm_up()
        except Exception as e:
            print(f"Warning: Failed to warm up worker on {self.device}: {e}")

class GPUWorkerPool:
    """Manages multiple GPU workers for parallel frame processing."""
    def __init__(
        self,
        devices: List[str],
        pipelines: Dict[str, TransformationPipeline],
        batch_size: int = 1
    ):
        """Initialize GPU worker pool.
        
        Args:
            devices: List of GPU devices
            pipelines: Dictionary mapping devices to pipelines
            batch_size: Frames to process per batch
        """
        self.devices = devices
        self.workers = []
        self.executor = None
        self.main_pbar = None
        self.gpu_pbars = None
        
        # Create workers
        for device in devices:
            if device not in pipelines:
                raise ValueError(f"No pipeline configured for device {device}")
            worker = GPUWorker(
                device=device,
                pipeline=pipelines[device],
                pool=self,
                batch_size=batch_size
            )
            self.workers.append(worker)
            
    def process_frames(
        self,
        input_dir: str,
        output_dir: str,
        interrupt_flag: Optional[Event] = None
    ) -> Dict[str, WorkerResult]:
        """Process frames using all available GPU workers.
        
        Args:
            input_dir: Directory containing input frames
            output_dir: Directory to save processed frames
            interrupt_flag: Optional interrupt flag
            
        Returns:
            Dictionary mapping devices to their WorkerResults
        """
        os.makedirs(output_dir, exist_ok=True)
        frames = sorted(os.listdir(input_dir))
        chunks = self._split_frames(frames)
        results: Dict[str, WorkerResult] = {}
        
        # Create progress bars
        self.main_pbar = tqdm(frames, desc="Total Progress", position=0)
        self.gpu_pbars = {
            device: tqdm(
                total=100,  # Use percentage for processor progress
                desc=f"GPU {device}: Idle",
                position=i+1,
                dynamic_ncols=True
            )
            for i, device in enumerate(self.devices)
        }

        try:
            self.executor = ThreadPoolExecutor(max_workers=len(self.devices))
            futures = {}
            
            # Submit work for each GPU
            for worker, chunk in zip(self.workers, chunks):
                future = self.executor.submit(
                    worker.process_frames,
                    chunk,
                    input_dir,
                    output_dir,
                    interrupt_flag
                )
                futures[future] = worker.device
                
            # Process results as they complete
            for future in as_completed(futures):
                device = futures[future]
                try:
                    result = future.result()
                    results[device] = result
                except Exception as e:
                    results[device] = WorkerResult(
                        device=device,
                        processed_frames=[],
                        failed_frames=chunks[self.devices.index(device)],
                        peak_memory=0,
                        avg_frame_time=0,
                        error=e
                    )
                    
        finally:
            # Clean up progress bars
            self.main_pbar.close()
            for pbar in self.gpu_pbars.values():
                pbar.close()
                
        return results
    
    def shutdown(self):
        """Terminate all workers and cleanup resources."""
        print("\nInitiating shutdown...")
        
        # Cancel any pending futures
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        
        # Cleanup GPU resources for each worker
        for worker in self.workers:
            try:
                with torch.cuda.device(worker.device):
                    torch.cuda.empty_cache()
                if hasattr(worker, 'pipeline'):
                    worker.pipeline.clean_up()
            except Exception as e:
                print(f"Error cleaning up worker on {worker.device}: {e}")
        self.workers.clear()
        print("Shutdown complete")

    def _split_frames(self, frames: List[str]) -> List[List[str]]:
        """Split frames among workers, accounting for GPU capabilities.
        
        Args:
            frames: List of frame filenames
            
        Returns:
            List of frame chunks for each worker
        """
        if not frames:
            return [[] for _ in self.devices]
        chunks = []
        chunk_size = len(frames) // len(self.devices)
        for i in range(len(self.devices)):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(self.devices) - 1 else None
            chunks.append(frames[start_idx:end_idx])
        return chunks