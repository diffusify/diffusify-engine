# setup.py
from setuptools import setup, find_packages

setup(
    name="diffusify-engine",
    version="0.1.1",
    packages=find_packages(where="src", include=["diffusify_engine*", "api*"]),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "ffmpeg-python>=0.2.0",
        "spandrel>=0.1.4"
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
