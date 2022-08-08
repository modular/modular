##=== setup.py ------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

from pathlib import Path

from setuptools import find_namespace_packages, setup

name_space = {}
with open(Path("modular") / "utils" / "version.py") as f:
    exec(f.read(), name_space)

setup(
    version=name_space["__version__"],
    packages=find_namespace_packages(exclude=["tests"]),
    zip_safe=False,
)
