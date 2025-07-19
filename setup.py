"""
Setup script for Clickbait Detection project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clickbait-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Clickbait Detection using Prompting and Fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/prompting-finetune",
    packages=find_packages(),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clickbait-experiment=scripts.run_experiment:main",
            "clickbait-prompting=src.experiments.pretrained_prompting:main",
            "clickbait-finetune=src.experiments.finetuning_experiment:main",
        ],
    },
)
