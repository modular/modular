# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn get_sliding_window_out_dim(
    in_dim: Int, ft_dim: Int, dilation: Int, stride: Int
) -> Int:
    """
    Return output dimension for a sliding window operation along some dimension.

    Args:
        in_dim: The size of the input dimension.
        ft_dim: The size of the corresponding filter dimension.
        dilation: The dilation for the sliding window operation.
        stride: The stride for the sliding window operation.

    Returns:
        The size of the output dimension.

    """
    return 1 + (in_dim - (1 + dilation * (ft_dim - 1))) // stride
