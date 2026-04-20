"""
NeuronBrain - Biological Neural Network Simulator
Setup configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuronbrain",
    version="1.0.0",
    author="NeuronBrain Team",
    author_email="contact@neuronbrain.org",
    description="Biological Neural Network Simulator with Hodgkin-Huxley, LIF, and Izhikevich neuron models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moggan1337/NeuronBrain",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "visualization": [
            "matplotlib>=3.4.0",
        ],
    },
    keywords=[
        "neuroscience",
        "neural-network",
        "spiking-neurons",
        "simulation",
        "hodgkin-huxley",
        "stdp",
        "computational-neuroscience",
    ],
    project_urls={
        "Bug Reports": "https://github.com/moggan1337/NeuronBrain/issues",
        "Source": "https://github.com/moggan1337/NeuronBrain",
        "Documentation": "https://github.com/moggan1337/NeuronBrain#readme",
    },
)
