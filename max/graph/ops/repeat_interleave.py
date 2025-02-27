# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for repeat_interleave."""

from typing import Optional

from ..value import Shape, TensorValue, TensorValueLike
from .reshape import reshape
from .tile import tile
from .unsqueeze import unsqueeze


def repeat_interleave(
    x: TensorValueLike,
    repeats: int,
    dim: Optional[int] = None,
) -> TensorValue:
    """Repeats elements of a tensor along the given dimension.

    Modeled after `torch.repeat_interleave`, with the constraint that
     Tensor-valued `repeats` are not yet supported.

    For example, given `repeats=2` and the following input:

    ```mojo
    input = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(2, 2),
        1.0, 2.0,
        3.0, 4.0,
    )
    ```

    `repeat_interleave` with `dim=0`:

    ```mojo
    output = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(4, 2),
        1.0, 2.0,
        1.0, 2.0,
        3.0, 4.0,
        3.0, 4.0,
    )
    ```

    `repeat_interleave` with `dim=1`:

    ```mojo
    output = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(2, 4),
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
    )
    ```

    `repeat_interleave` with `dim=None` (the default):

    ```mojo
    output = max.tensor.Tensor[DType.float32](
        max.tensor.TensorShape(8),
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0,
    )
    ```

    Args:
        input: The input tensor.
        repeats: The number of repetitions for each element.
        dim: The dimension along which to repeat values. By default (or if `dim`
             is `None`), flatten the input array.

    Returns:
        A symbolic tensor with the elements interleaved.
    """
    if repeats <= 0:
        raise ValueError(
            f"repeats for repeat_interleave has to be >= 0. Got {repeats}."
        )

    x = TensorValue(x)
    if dim is not None and (dim < 0 or dim > x.rank):
        raise ValueError(
            f"dim for repeat_interleave on tensor x has to be between 0 and {x.rank}. Got {dim}."
        )

    # For compatibility with Torch, if `dim` is not passed, flatten the input array and return a flat array.
    if dim is None:
        x = x.reshape([-1])
        dim = 0

    # To implement `repeat_interleave`, we need to unsqueeze at dim+1 so that
    # we can tile each element of the given dim. (Tiling without unsqueezing
    # would not give us interleaved elements.) We can then squeeze the input
    # back to the expected shape. For example: input=1x8x1025x128xbf16,
    # repeats=3, and dim=1 would be:
    #    - input:     1x8x1025x128
    #    - unsqueeze: 1x8x1x1025x128   (use dim+1)
    #    - tile:      1x8x3x1025x128   (use dim+1)
    #    - reshape:   1x24x1025x128
    # The one exception is when `dim` is `None`, in which case the input array
    # is flattened and all elements are tiled.
    tile_dim = dim + 1
    # Tile by repeats on the tile dimension
    tiles = [1] * x.rank
    tiles.insert(tile_dim, repeats)

    # Multiply by repeats on dim
    result_shape = Shape(x.shape)
    result_shape[dim] *= repeats

    unsqueezed = unsqueeze(x, tile_dim)
    tiled = tile(unsqueezed, tiles)
    return reshape(tiled, result_shape)
