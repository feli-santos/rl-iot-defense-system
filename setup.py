from setuptools import setup, find_packages

setup(
    name="iotdefense",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=1.0.0",
        "stable-baselines3>=2.5.0",
        "torch>=2.6.0",
        "numpy>=2.2.0",
        "pyyaml>=6.0",
        "matplotlib>=3.10.0",
        "tensorboard>=2.19.0",
    ],
    python_requires=">=3.12",
    author="Felipe Santos",
    author_email="your.email@example.com",
    description="RL-based IoT Defense System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rl-iot-defense-system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)