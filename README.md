# Diffusify Engine
**_Create and Enhance Videos with AI_**


Diffusify Engine is a powerful, GPU-accelerated system designed to process and create videos using the latest AI models. It leverages the power of multiple GPUs to deliver high performance and speed.

### What is Diffusify Engine?

Diffusify Engine is a modular video processing and generation framework built around the concept of customizable pipelines. These pipelines are composed of individual processing units, referred to as "processors," each designed to execute a specific operation on video data.

Processors can encapsulate a variety of functionalities, including generative frame generation, upscaling, denoising, artifact removal, segmentation, frame interpolation, and more. Each processor can be configured independently, and the pipeline can be distributed across multiple GPUs for faster processing.

## Key Features

- **Customizable Processing Pipelines**: Create complex processing sequences by chaining together multiple processors.
- **Multi-GPU Support**: Automatically distribute the workload across multiple GPUs for improved performance.
- **Scalable Tiling Mechanism**: Efficiently process high-resolution frames using tiling to manage GPU memory constraints.
- **Graceful Shutdown and Interrupt Handling**: Safely handle termination signals to prevent data loss and ensure proper resource cleanup.
- **Extensible Framework**: Easily add new processors by inheriting from base classes.
- **Text-to-Video Frame Generation**: Generate video frames from text prompts using configurable generative models.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Transformation Mode](#transformation-mode)
  - [Generative Mode](#generative-mode)
  - [Configuring Pipelines](#configuring-pipelines)
  - [Using Multiple GPUs](#using-multiple-gpus)
- [Adding Custom Processors](#adding-custom-processors)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- **Python 3.10 or higher**
- **PyTorch**: For GPU computations.
- **ffmpeg**: For video handling.
- **CUDA-compatible GPUs**: For leveraging GPU acceleration.
- Additional Python packages as listed in `requirements.txt`.

## Installation (Linux)

### 1. Clone the Repository

```bash
git clone https://github.com/diffusify/diffusify-engine.git
cd diffusify-engine
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended), and install dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install ffmpeg

Ensure `ffmpeg` is installed and available in your system PATH.

- **Ubuntu/Debian**:

  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```

- **Windows**:

  Download the static build from the [ffmpeg website](https://ffmpeg.org/download.html#build-windows) and add it to your PATH.

**Note: This application has not been tested on Windows or MacOS. (planned for future releases)**

### 4. Set Up Model Weights

Download the required model weights and place them in the appropriate directory. For example, for the `SpandrelProcessor`:

- Create a `weights` directory in the project root.

  ```bash
  mkdir weights
  ```

- Place your `.safetensors` model files in the `weights` directory.

### 5. Include the Spandrel Library

The Spandrel library is required for certain processors like `SpandrelProcessor`.

## Project Structure

```
diffusify-engine/
├── src/
│   ├── main.py                             # Command-line interface entry point (for testing/dev).
│   ├── api/                                # Web API layer (Placeholder - not implemented in provided code)
│   │   ├── app.py                          # Flask/FastAPI application setup.
│   │   └── routes/                         # API endpoint definitions.
│   ├── diffusify_engine/                   # Core logic of the application
│   │   ├── generative/                     # Experimental text-to-video, image-to-video, video-to-video support (wip)
│   │   ├── transformation_manager.py       # Manages the video transformation workflow.
│   │   ├── transformation/                 # Contains transformation-related classes
│   │   │   ├── gpu_pool.py                 # Manages GPU workers for transformations.
│   │   │   ├── transformation_pipeline.py  # Defines the processing pipeline for transformations.
│   │   │   └── processors/                 # Implementations of different image processing transforms
│   │   │       ├── spandrel.py             # Implements a processor using the Spandrel model library.
│   │   │       └── base.py                 # Abstract base class for all processors.
│   │   ├── utils/                          # Utility functions
│   │   │   ├── tensor_processor.py         # Implements tensor processing functions like tiled scaling and gaussian blur.
│   │   │   ├── timing.py                   # Provides timing utilities for measuring execution time.
│   │   │   ├── stats.py                    # Formats processing statistics into a human-readable string.
│   │   │   └── gpu.py                      # Provides GPU related utility functions.
├── weights/
│   └── [model files]
├── requirements.txt
└── setup.py
```

## Usage

### Quick Start

The `main.py` script now supports two modes: `transformation` for video processing and `generative` for text-to-video frame generation.

### Transformation Mode

To process a video using the default settings:

```bash
python src/main.py --mode transformation -i input_video.mp4 -o output_video.mp4
```

This will use the default processing pipeline, which includes the `SpandrelProcessor` with default settings.

### Generative Mode

To generate a video from a text prompt:

```bash
python src/main.py --mode generative -p "A majestic lion walking through a savanna at sunset" -o generated_video.mp4
```

This will use the configured generative processor to create video frames based on the provided prompt.

### Configuring Pipelines

To customize the processing pipeline, you can either modify the default pipeline configuration in `transformation_manager.py` or provide a custom configuration when initializing the `TransformationManager` class directly (not shown in the quick start). The configuration is a list of dictionaries, where each dictionary defines a processor.

```python
pipeline_config = [
    {
        'type': 'spandrel',
        'name': '4x_upscaler',
        'model_path': 'weights/4xRealWebPhoto-v4-dat2.safetensors',
        'config': {
            'tile_size': 512,
            'min_tile_size': 128,
            'overlap': 32
        }
    },
    # Add more processors as needed
]

# Specify devices (e.g., "cuda:0" or "cuda:0,cuda:1")
devices = "cuda:0"

processor = TransformationManager(devices=devices, pipeline_config=pipeline_config)
```

### Using Multiple GPUs

To leverage multiple GPUs, list them in the `devices` parameter.

```python
devices = "cuda:0,cuda:1"

processor = TransformationManager(devices=devices, pipeline_config=pipeline_config)
```

The `GPUWorkerPool` will automatically distribute frames among the specified GPUs.

## Adding Custom Processors

To extend the processing capabilities, you can create custom processors by inheriting from the `BaseProcessor` class and implementing the `process` and `get_scale_factor` methods. For generative models, implement the `generate` method.

## Performance Tips

- **Optimize Tile Sizes**: Adjust `tile_size`, `min_tile_size`, and `overlap` in the processor configuration to balance between performance and memory usage.
- **Monitor GPU Memory**: Use tools like `nvidia-smi` to monitor GPU memory usage and adjust configurations accordingly.
- **Use Appropriate Batch Sizes**: While the provided code processes frames individually, batch processing can be implemented for models that support it.
- **Warm-Up Models**: The pipeline includes a warm-up step to pre-load models onto the GPU, reducing initialization overhead during processing.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

1. **Fork** the repository.
2. **Create** a new branch with a descriptive name.
3. **Make** your changes.
4. **Submit** a pull request to the `main` branch.

Please ensure that your code follows the existing style and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
