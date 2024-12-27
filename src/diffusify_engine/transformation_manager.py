import os
import sys
import shutil
import time
import subprocess
import traceback
import signal
import threading
import torch
import ffmpeg

from typing import Optional, Dict, List, Any

from .transformation.gpu_pool import GPUWorkerPool, WorkerResult
from .transformation.transformation_pipeline import TransformationPipeline
from .transformation.processors.spandrel import SpandrelProcessor
from .utils import parse_gpu_devices, timed_execution, format_stats_output

class TransformationManager:
    def __init__(
        self,
        devices: Optional[str] = None,
        pipeline_config: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize the video processor with processing pipeline.
        
        Args:
            devices: GPU devices to use (e.g. "cuda:0,cuda:1")
            pipeline_config: List of processor configurations
        """
        # Initialize models and devices
        self.interrupt_flag = threading.Event()
        self.devices = parse_gpu_devices(devices)
        self._setup_signal_handlers()

        # Initialize processing pipelines for each GPU
        self.pipelines = self._create_pipelines(pipeline_config or self._default_pipeline_config())

        # Initialize GPU worker pool
        self.worker_pool = GPUWorkerPool(
            devices=self.devices,
            pipelines=self.pipelines
        )

        # Initialize stats dictionary
        self.stats = {
            'input_file': None,
            'output_file': None,
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'metadata': {},
            'timings': {
                'extraction': None,
                'processing': None,
                'encoding': None
            },
            'processing_stats': {
                'total_frames': None,
                'processed_frames': 0,
                'failed_frames': [],
                'avg_frame_time': None,
                'gpu_memory_peak': None if torch.cuda.is_available() else 'N/A',
                'gpu_stats': {} 
            }
        }

    def _default_pipeline_config(self) -> List[Dict[str, Any]]:
        """Create default pipeline configuration.
        
        Returns:
            List of processor configurations
        """
        return [
        {
            'type': 'spandrel',
            'name': '4xNomos8kDAT',
            'model_path': 'weights/4xNomos8kDAT.safetensors',
            'config': {
                'use_amp': False,
                'compile_model': False,
                'tile_size': 768,
                'min_tile_size': 128,
                'overlap': 32
            }
        }
        # {
        #     'type': 'spandrel',
        #     'name': '4xRealWebPhoto-v4-dat2',
        #     'model_path': 'weights/4xRealWebPhoto-v4-dat2.safetensors',
        #     'config': {
        #         'use_amp': False,
        #         'compile_model': False,
        #         'tile_size': 512,
        #         'min_tile_size': 128,
        #         'overlap': 32
        #     }
        # }
        ]

    def _create_pipelines(
        self,
        pipeline_config: List[Dict[str, Any]]
    ) -> Dict[str, TransformationPipeline]:
        """Create processing pipelines for each GPU.
        
        Args:
            pipeline_config: List of processor configurations
            
        Returns:
            Dictionary mapping device to pipeline
        """
        pipelines = {}
        
        for device in self.devices:
            processors = []
            for proc_config in pipeline_config:
                if proc_config['type'] == 'spandrel':
                    processor = SpandrelProcessor(
                        name=proc_config['name'],
                        model_path=proc_config['model_path'],
                        device=device,
                        config=proc_config.get('config')
                    )
                    processors.append(processor)

            pipelines[device] = TransformationPipeline(
                processors=processors,
                output_device=device
            )
        return pipelines
    
    def extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffmpeg."""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            return {
                "width": int(video_info['width']),
                "height": int(video_info['height']),
                "fps": eval(video_info['avg_frame_rate']),
                "total_frames": int(video_info['nb_frames']),
                "has_audio": audio_info is not None
            }
        except Exception as e:
            print(f"Error extracting video metadata: {e}")
            sys.exit(1)

    def extract_frames(self, video_path: str, output_dir: str) -> None:
        """Extract frames from video using ffmpeg."""
        os.makedirs(output_dir, exist_ok=True)
        fps = self.stats['metadata']['fps']
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-qscale:v', '1', '-qmin', '1', '-qmax', '1',
                ###
                # '-vframes', '1',  # Temporary limit frames
                ###
                '-r', f'{fps}', f'{output_dir}/frame%08d.png'
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frames: {e}")
            traceback.print_exc()
            sys.exit(1)

    def process_frames(self, input_dir: str, output_dir: str) -> None:
        """Process frames using GPU worker pool."""
        try:
            # Get frame list and update total frames count
            frames = sorted(os.listdir(input_dir))
            self.stats['processing_stats']['total_frames'] = len(frames)
            
            print(f"\nProcessing {len(frames)} frames using {len(self.devices)} GPUs...")
            
            # Process frames using worker pool
            results: Dict[str, WorkerResult] = self.worker_pool.process_frames(
                input_dir=input_dir,
                output_dir=output_dir,
                interrupt_flag=self.interrupt_flag
            )

            # Aggregate results and update stats
            total_processed = 0
            failed_frames = []
            
            for device, result in results.items():
                # Update processed/failed frame counts
                total_processed += len(result.processed_frames)
                failed_frames.extend(result.failed_frames)
                
                # Store GPU-specific stats
                self.stats['processing_stats']['gpu_stats'][device] = {
                    'processed_frames': len(result.processed_frames),
                    'failed_frames': len(result.failed_frames),
                    'peak_memory': result.peak_memory,
                    'avg_frame_time': result.avg_frame_time
                }
                
                # Log any device-specific errors
                if result.error:
                    print(f"\nError on {device}: {str(result.error)}")
            
            # Update global stats
            self.stats['processing_stats']['processed_frames'] = total_processed
            self.stats['processing_stats']['failed_frames'] = failed_frames
            
            # Calculate overall average frame time
            if total_processed > 0:
                total_time = sum(
                    stats['avg_frame_time'] * stats['processed_frames']
                    for stats in self.stats['processing_stats']['gpu_stats'].values()
                    if stats['avg_frame_time'] is not None
                )
                self.stats['processing_stats']['avg_frame_time'] = \
                    total_time / total_processed
            
        except Exception as e:
            print(f"Error processing frames: {e}")
            traceback.print_exc()
            raise
        finally:
            # Ensure worker pool is shut down
            self.worker_pool.shutdown()

    def create_video(
        self, 
        frame_dir: str, 
        output_path: str, 
        fps: float,
        original_video: str
    ) -> None:
        """Create video from processed frames."""
        try:
            # Setup video encoder
            video_stream = ffmpeg.input(f'{frame_dir}/frame%08d.png', r=fps)
            
            # If original video has audio, include it
            if self.stats['metadata']['has_audio']:
                audio_stream = ffmpeg.input(original_video).audio
                stream = ffmpeg.output(
                    video_stream, 
                    audio_stream,
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    pix_fmt='yuv420p',
                    crf=12
                )
            else:
                stream = ffmpeg.output(
                    video_stream,
                    output_path,
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    crf=12
                )
            
            stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
        except ffmpeg.Error as e:
            print(f"Error creating video: {e.stderr.decode()}")
            traceback.print_exc()
            sys.exit(1)

    def process_video(
            self,
            input_path: str,
            output_path: str,
            temp_dir: Optional[str] = None
        ) -> None:
            """Process a video file using the configured pipeline.
            
            Args:
                input_path: Path to input video file
                output_path: Path where output video will be saved
                temp_dir: Optional temporary directory for processing
            """
            # Initialize timing stats
            self.stats['input_file'] = input_path
            self.stats['output_file'] = output_path
            self.stats['start_time'] = time.time()

            try:
                # Create temp directories
                temp_dir = temp_dir or f"tmp_{int(time.time())}"
                frame_dir = os.path.join(temp_dir, "frames")
                processed_dir = os.path.join(temp_dir, "processed")
                os.makedirs(temp_dir, exist_ok=True)

                # Extract video metadata
                self.stats['metadata'] = self.extract_metadata(input_path)

                # Warm-up pipelines
                # self._warm_up_pipelines()
                
                # Extract frames with timing
                print("\nExtracting frames...")
                timed_execution(
                    self.stats,
                    lambda: self.extract_frames(input_path, frame_dir),
                    'extraction'
                )

                # Process frames with timing
                print("\nProcessing frames...")
                timed_execution(
                    self.stats,
                    lambda: self.process_frames(frame_dir, processed_dir),
                    'processing'
                )

                # Create output video with timing
                if not self.interrupt_flag.is_set():
                    print("\nCreating output video...")
                    timed_execution(
                        self.stats,
                        lambda: self.create_video(
                            processed_dir, 
                            output_path,
                            self.stats['metadata']['fps'],
                            input_path
                        ),
                        'encoding'
                    )

            except Exception as e:
                print(f"\nError processing video: {e}")
                traceback.print_exc()
                raise
                
            finally:
                # Update end time
                self.stats['end_time'] = time.time()
                self.stats['total_duration'] = self.stats['end_time'] - self.stats['start_time']
                
                # Cleanup
                self._cleanup(temp_dir)
                
                # Print final stats
                print(format_stats_output(self.stats))

    def _warm_up_pipelines(self):
       """Warm up all processing pipelines."""
       print("\nWarming up processing pipelines...")
       for device, pipeline in self.pipelines.items():
           try:
               pipeline.warm_up()
           except Exception as e:
               print(f"Warning: Failed to warm up pipeline on {device}: {e}")

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def signal_handler(frame, other):
            print("\nInterrupt received, initiating graceful shutdown...")
            self.interrupt_flag.set()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _cleanup(self, temp_dir: Optional[str]):
        """Cleanup resources and temporary files."""
        try:
            # Shutdown worker pool
            if hasattr(self, 'worker_pool'):
                self.worker_pool.shutdown()
            
            # Remove temp directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
            print("\nCleanup completed successfully")
            
        except Exception as e:
            print(f"\nError during cleanup: {e}")