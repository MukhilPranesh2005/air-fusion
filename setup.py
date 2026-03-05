"""
Setup script for Multi-Modal Air Quality Prediction System

Author: Air Quality Prediction System
Date: 2026
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="air-quality-prediction",
    version="1.0.0",
    author="Air Quality Prediction System",
    author_email="contact@airquality.ai",
    description="Multi-modal deep learning system for air quality prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/air-quality-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.24.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "air-quality-generate=data_generator:main",
            "air-quality-train=train:main",
            "air-quality-predict=inference:main",
            "air-quality-evaluate=evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="air quality prediction deep learning multi-modal tensorflow",
    project_urls={
        "Bug Reports": "https://github.com/your-username/air-quality-prediction/issues",
        "Source": "https://github.com/your-username/air-quality-prediction",
        "Documentation": "https://github.com/your-username/air-quality-prediction/blob/main/README.md",
    },
)
