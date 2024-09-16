# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for load_buffer."""

import sys
from typing import TYPE_CHECKING, Union

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

if TYPE_CHECKING:
    # EllipsisType was added in 3.10, but we support down to 3.9.
    # Make this import unconditional when we drop 3.9 (MSDK-756).
    from types import EllipsisType

import numpy as np
from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import BufferType, Dim, DimLike, Shape, StaticDim, TensorType
from ..value import BufferValue, TensorValue
from .constant import constant
from .select import select
from .stack import stack_scalars


def load_buffer(
    x: TensorValue,
) -> BufferValue:
    # TODO: implement load_buffer through rmo.mo_mutable.load
    raise NotImplementedError("TODO")
