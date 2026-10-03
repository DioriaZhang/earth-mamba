from setuptools import setup, find_packages

setup(
    name="earth-mamba",
    version="0.1.0",
    packages=find_packages(include=["earth_mamba", "earth_mamba.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "triton>=3.0",
        "einops",
        "timm",
    ],
)
