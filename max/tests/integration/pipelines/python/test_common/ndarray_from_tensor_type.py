# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Functions for converting graph tensor types to runtime ndarrays."""

import numpy as np
import numpy.typing as npt
from max.graph import Dim, StaticDim, SymbolicDim, TensorType


def _static_dim_from_dim(
    dim: Dim, name_to_static_dim: dict[str, int], max_dim_size: int
) -> int:
    """Creates random ints for symbolic dims and just returns static dims."""
    if isinstance(dim, SymbolicDim):
        if dim.name not in name_to_static_dim:
            # Bind the symbolic dim to an integer value by setting the name to
            # static map, if not already set.
            name_to_static_dim[dim.name] = np.random.randint(max_dim_size)

        return name_to_static_dim[dim.name]

    if isinstance(dim, StaticDim):
        # Return the dim since it's already static.
        return int(dim)

    raise TypeError("algebraic dims unsupported in ndarray_from_tensor_type")


def ndarray_from_tensor_type(
    *tensor_types: TensorType, max_dim_size: int = 128
) -> list[npt.NDArray]:
    """Creates a list of ndarrays from a variadic list of graph tensor types.

    This respects the static dimensions of dims that are already StaticDims.
    For SymbolicDims, this generates a random dimension for each unique symbolic dimension name.
    """
    name_to_static_dim: dict[str, int] = {}
    ndarrays: list[npt.NDArray] = []
    for tens_type in tensor_types:
        # Staticize the dims in the tensor type, updating the symbolic dim map.
        static_dims = [
            _static_dim_from_dim(dim, name_to_static_dim, max_dim_size)
            for dim in tens_type.shape
        ]

        # Randomly generate an ndarray with shape given by the static dims.
        ndarrays.append(
            np.random.randn(*static_dims).astype(tens_type.dtype.to_numpy())
        )

    return ndarrays
