# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from memory.memory import memset_zero
from runtime.llcl import Runtime

# Indicate position in pads tensor for height, width.
alias PADS_H_START = 0
alias PADS_H_END = 2
alias PADS_W_START = 1
alias PADS_W_END = 3


# TODO: All attributes, except for groups and auto_pad, are supported.
#       - Kernel assumes groups = 1.
#       - For auto_pad, need to set `AutoPadMode.NOTSET` (default).
#       Only remaining issue is handling of optional attributes & setting defaults,
#       and the associated logic (e.g., if one attribute is specified another is
#       ignored).
#       Specifically, in the ONNX spec for ConvTranspose
#       (https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose)
#       - dilations (optional): if not provided, create (1,1) as default.
#       - kernel_shape (optional): if not provided, obtain from argument-provided kernel.
#       - strides (optional): if not provided, create (1,1) as default.
#       - output_shape (optional): if specified, pads values are ignored.
#       modular/Kernels/test/test_convtranspose.mojo provides examples of calls.
#       StarGAN, CycleGAN-and-pix2pix, Mask-RCNN are covered by this version.


@always_inline
fn conv_transpose[
    rank: Int,
    type: DType,
    strides_type: DType,
    dilations_type: DType,
    pads_type: DType,
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    input: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        type,
    ],
    kernel: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        type,
    ],
    strides: NDBuffer[1, DimList.create_unknown[1](), strides_type],
    dilations: NDBuffer[1, DimList.create_unknown[1](), dilations_type],
    pads: NDBuffer[1, DimList.create_unknown[1](), pads_type],
):
    """
    Implements the ConvTranspose operator from the MO spec.

    Parameters:
        rank: Rank of the input, output, and kernel tensors.
        type: Type of the input, output, and kernel tensors.
        strides_type: Element type of strides.
        dilations_type: Element type of dilations.
        pads_type: Element type of pads.

    Args:
        output: Output data tensor that contains the result of the convolution.
        input: Input data tensor from previous layer, with size of (N x H x W x C),
               where N is the batch size, C is the number of channels, and H and
               W are the height and width.
        kernel: The weight (kernel) tensor, with size of (kH x kW x M/groups x C),
                where C is the number of channels, kH and kW are the height and
                width of the kernel, and M is the number of feature maps.
        strides: Stride along each spatial axis.
        dilations: Dilation value along each spatial axis of the filter.
        pads: Padding at the beginning and ending of each spatial axis. Follows
              the format [x1_begin, x2_begin, x1_end, x2_end].
    """

    let N = Int(input.dim(0))  # Number of images (num. batches)
    let H = Int(input.dim(1))  # Input height
    let W = Int(input.dim(2))  # Input width
    let C = Int(input.dim(3))  # Number of input channels

    let R = Int(kernel.dim(0))  # Filter height
    let S = Int(kernel.dim(1))  # Filter width
    let C_filter = Int(kernel.dim(2))  # Number of input channels

    let HO = Int(output.dim(1))
    let WO = Int(output.dim(2))

    # Initialize output to zero
    memset_zero[type](output.data, N * C_filter * HO * WO)

    for n in range(N):
        for c in range(C):
            for f in range(C_filter):
                for i in range(H):
                    let indX_out = i * int(strides[0]) - int(pads[PADS_H_START])
                    for j in range(W):
                        let indY_out = j * int(strides[1]) - int(
                            pads[PADS_W_START]
                        )
                        for r in range(R):
                            for s in range(S):
                                let x_out = indX_out + r * int(dilations[0])
                                let y_out = indY_out + s * int(dilations[1])
                                if (
                                    x_out >= 0
                                    and x_out < HO
                                    and y_out >= 0
                                    and y_out < WO
                                ):
                                    let tmp = output[n, x_out, y_out, f]
                                    output[
                                        StaticIntTuple[rank](n, x_out, y_out, f)
                                    ] = (
                                        tmp
                                        + input[n, i, j, c] * kernel[r, s, f, c]
                                    )


@always_inline
fn conv_transpose_shape[
    input_rank: Int,
    kernel_rank: Int,
    type: DType,
    strides_type: DType,
    dilations_type: DType,
    pads_type: DType,
    output_pads_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[
        input_rank,
        DimList.create_unknown[input_rank](),
        type,
    ],
    kernel: NDBuffer[
        kernel_rank,
        DimList.create_unknown[kernel_rank](),
        type,
    ],
    strides: NDBuffer[1, DimList.create_unknown[1](), strides_type],
    dilations: NDBuffer[1, DimList.create_unknown[1](), dilations_type],
    pads: NDBuffer[1, DimList.create_unknown[1](), pads_type],
    output_pads: NDBuffer[1, DimList.create_unknown[1](), output_pads_type],
) raises -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `conv-transpose` operation, and assert the
    inputs are compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        kernel_rank: Rank of the kernel tensor.
        type: Element type of the input and kernel tensor.
        strides_type: Element type of the strides tensor.
        dilations_type: Element type of the dilations tensor.
        pads_type: Element type of the pads tensor.
        output_pads_type: Element type of the output_pads tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input: The input tensor.
        kernel: The kernel tensor.
        strides: The strides tensor.
        dilations: The dilations tensor.
        pads: The paddings tensor.
        output_pads: The output paddings tensor.

    Returns:
        The output shape.
    """

    if input_rank != 4:
        raise Error("[conv_transpose] requires (input_rank == 4))")
    if input_rank != kernel_rank:
        raise Error("[conv_transpose] requires (input_rank == kernel_rank))")
    if strides.dim(0) != input_rank - 2 or dilations.dim(0) != input_rank - 2:
        raise Error(
            "[conv_transpose] requires (len(strides) == len(dilations) =="
            " input_rank - 2)"
        )
    if pads.dim(0) != 2 * (input_rank - 2):
        raise Error(
            "[conv_transpose] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume input has layout NHWC
    let batch_size = input.dim(0)
    let input_channels = input.dim(3)

    # Assume kernel has layout RSCF, the output channel is C because this is a
    # convolution transpose shape function (inverse of regular convolution).
    let output_channels = kernel.dim(2)

    # compute and return the output shape
    let output_height = (
        int(strides[0]) * (input.dim(1) - 1)
        + int(output_pads[0])
        + ((kernel.dim(0) - 1) * int(dilations[0]) + 1)
        - int(pads[PADS_H_START])
        - int(pads[PADS_H_END])
    )
    let output_width = (
        int(strides[1]) * (input.dim(2) - 1)
        + int(output_pads[1])
        + ((kernel.dim(1) - 1) * int(dilations[1]) + 1)
        - int(pads[PADS_W_START])
        - int(pads[PADS_W_END])
    )

    if output_height <= 0:
        raise Error("[conv_transpose] output height must be positive")
    if output_width <= 0:
        raise Error("[conv_transpose] output width must be positive")

    var output_shape = StaticIntTuple[input_rank]()
    output_shape[0] = batch_size
    output_shape[1] = output_height
    output_shape[2] = output_width
    output_shape[3] = output_channels

    return output_shape
