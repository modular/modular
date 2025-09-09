# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Experimental APIs for the MAX platform."""

from . import functional, random, tensor
from .tensor import Tensor

__all__ = ["Tensor", "functional", "random", "tensor"]
