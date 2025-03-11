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
    x: TensorValueLike, repeats: int, axis: Optional[int] = None
) -> TensorValue:
    """Repeats elements of a tensor along the given dimension.

    Modeled after :obj:`torch.repeat_interleave`, with the constraint that
    Tensor-valued ``repeats`` are not yet supported.

    For example, given ``repeats=2`` and the following input:

    .. code-block:: python

        # Input tensor with shape (2, 2)
        input = TensorValue(x)  # Contains [[1.0, 2.0], [3.0, 4.0]]

    ``repeat_interleave`` with ``axis=0``:

    .. code-block:: python

        # Output tensor with shape (4, 2)
        output = repeat_interleave(input, repeats=2, axis=0)
        # Contains [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]

    ``repeat_interleave`` with ``axis=1``:

    .. code-block:: python

        # Output tensor with shape (2, 4)
        output = repeat_interleave(input, repeats=2, axis=1)
        # Contains [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]

    ``repeat_interleave`` with ``axis=None`` (the default):

    .. code-block:: python

        # Output tensor with shape (8,)
        output = repeat_interleave(input, repeats=2)  # axis = None
        # Contains [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]

    Args:
        x:
            The input tensor.
        repeats:
            The number of repetitions for each element.
        axis:
            The dimension along which to repeat values. If axis is not
            specified or None (the default), flatten the input array
            and repeat the flattened values.

    Returns:
        A symbolic tensor with the elements interleaved.

    Raises:
        ValueError: If ``repeats`` non-positive or if ``axis`` is out of range.
    """
    x = TensorValue(x)

    if repeats <= 0:
        raise ValueError(f"repeat_interleave: {repeats=} must be > 0")

    if axis is not None and not -x.rank <= axis < x.rank:
        raise ValueError(
            f"repeat_interleave: {axis=} out of bounds for {x.rank=}"
        )

    # For compatibility with Torch, if `axis` is not passed, flatten the input array and return a flat array.
    if axis is None:
        x = x.reshape([-1])
        axis = 0

    if axis < 0:
        axis += x.rank

    # To implement `repeat_interleave`, we need to unsqueeze at axis+1 so that
    # we can tile each element of the given axis. (Tiling without unsqueezing
    # would not give us interleaved elements.) We can then squeeze the input
    # back to the expected shape. For example: input=1x8x1025x128xbf16,
    # repeats=3, and axis=1 would be:
    #    - input:     1x8x1025x128
    #    - unsqueeze: 1x8x1x1025x128   (use axis+1)
    #    - tile:      1x8x3x1025x128   (use axis+1)
    #    - reshape:   1x24x1025x128
    # The one exception is when `axis` is `None`, in which case the input array
    # is flattened and all elements are tiled.
    tile_dim = axis + 1
    # Tile by repeats on the tile dimension
    tiles = [1] * x.rank
    tiles.insert(tile_dim, repeats)

    # Multiply by repeats on axis
    result_shape = Shape(x.shape)
    result_shape[axis] *= repeats

    unsqueezed = unsqueeze(x, tile_dim)
    tiled = tile(unsqueezed, tiles)
    return reshape(tiled, result_shape)
