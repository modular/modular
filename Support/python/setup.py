# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

from setuptools import find_namespace_packages, setup


def get_version() -> str:
    """Get the version of the utility library.

    Returns:
        The version as a string.
    """
    name_space = {}
    with open(Path("modular") / "utils" / "version.py") as f:
        exec(f.read(), name_space)
    return name_space["__version__"]


README_PATH = Path(__file__).parent / "README.md"

setup(
    name="modular-pyutils",
    version=get_version(),
    author="Modular",
    license="Modular Proprietary",
    description="Modular Python utilities",
    long_description=README_PATH.read_text(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(exclude=["tests"]),
    zip_safe=False,
    python_requires=">=3.8",
    # WARNING: if changing dependencies, need to bump the version as well.
    install_requires=[
        "numpy",
        "ruamel.yaml>=0.17",
        "find-libpython>=0.3.0",
    ],
    extras_require={
        "tests": [
            "pytest>=7.1.2",
            "pytest-cov",
        ],
        "dev": [
            "black>=22.12.0",
            "isort>=5.10.1",
            "flake8>=3.9",
            "pyright>=1.1.255",
        ],
        "build-metrics": [
            "opentelemetry-api>=1.18.0",
            "opentelemetry-sdk>=1.18.0",
            "opentelemetry-exporter-otlp>=1.18.0",
        ],
    },
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
