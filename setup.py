# setup.py
from setuptools import setup, find_packages

setup(
    name="diffusify-engine",
    version="0.1.1",
    packages=find_packages(where="src", include=["diffusify_engine*", "api*"]),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.5.1",
        "accelerate==1.2.1",
        "diffusers==0.32.0",
        "transformers==4.47.1",
        "tqdm",
        "safetensors",
        "torchvision",
        "ffmpeg-python",
        "spandrel"
    ],
    extras_require={
        "dev": [
            "pytest",
            # ... other ...
        ]
    },
    entry_points={
        "console_scripts": [
            "diffusify=diffusify_engine.main:main",
        ],
    },
)
