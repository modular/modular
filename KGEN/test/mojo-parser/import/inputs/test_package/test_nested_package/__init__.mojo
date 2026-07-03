# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .module import nested_function, parametric_fn

# Re-export the public submodules so they are reachable as attributes
# (`test_nested_package.module`, `test_nested_package.deep_package`).
from . import module
from . import deep_package
