#!/usr/bin/env python3
"""Setup script for gemma-finetune-emails package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gemma-finetune-emails",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A clean, modular pipeline for fine-tuning Google's Gemma 2B model using LoRA on email intent classification tasks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gemma-finetune-emails",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gemma-finetune-emails/issues",
        "Source": "https://github.com/yourusername/gemma-finetune-emails",
        "Documentation": "https://github.com/yourusername/gemma-finetune-emails#readme",
    },
    packages=find_packages(include=["src", "src.*", "configs", "configs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gemma-finetune=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning", 
        "nlp", 
        "lora", 
        "fine-tuning", 
        "email-classification", 
        "gemma", 
        "transformers"
    ],
)
