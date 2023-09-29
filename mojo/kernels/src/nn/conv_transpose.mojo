# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from runtime.llcl import Runtime, OutputChainPtr

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
#       - output_padding (optional): if not provided, create (0,0) as default.
#       - strides (optional): if not provided, create (1,1) as default.
#       - bias (optional): currently, we use bias_add parameter to use or ignore.
#       - output_shape (optional): if specified, pads values are ignored.
#       modular/Kernels/test/test_convtranspose.mojo provides examples of calls.
#       StarGAN, CycleGAN-and-pix2pix, Mask-RCNN are covered by this version.


@always_inline
fn convtranspose[
    rank: Int,
    type: DType,
    epilogue_fn: fn (Int, SIMD[type, 1]) capturing -> SIMD[type, 1],
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
    strides: NDBuffer[1, DimList.create_unknown[1](), DType.index],
    dilations: NDBuffer[1, DimList.create_unknown[1](), DType.index],
    pads: NDBuffer[1, DimList.create_unknown[1](), DType.index],
    output_padding: NDBuffer[1, DimList.create_unknown[1](), DType.index],
    out_chain: OutputChainPtr,
):
    """
    Implements the ConvTranspose operator from the ONNX spec:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
    Computes cumulative sum of the input elements along the given axis.
    Cumulative sum can be inclusive or exclusive of the top element, and
    normal or reverse (direction along a given axis).

    Parameters:
        rank: Rank of the input, output, and kernel tensors.
        type: Type of the input, output, and kernel tensors.
        epilogue_fn: Epilogue function used to calculate bias.

    Args:
        output: Output data tensor that contains the result of the convolution.
        input: Input data tensor from previous layer, with size of (N x C x H x W),
               where N is the batch size, C is the number of channels, and H and
               W are the height and width.
        kernel: The weight (kernel) tensor, with size of (N x M/group x kH x kW),
                where C is the number of channels, kH and kW are the height and
                width of the kernel, and M is the number of feature maps.
        strides: Stride along each spatial axis.
        dilations: Dilation value along each spatial axis of the filter.
        pads: Padding at the beginning and ending of each spatial axis. Follows
              the format [x1_begin, x2_begin, x1_end, x2_end].
        output_padding: Additional elements added to the side with higher
                        coordinate indices in the output.
        out_chain: The OutputChainPtr used to mark competion or error of the task.
    """

    let N = Int(input.dim(0))  # Number of images (num. batches)
    let H = Int(input.dim(2))  # Input height
    let W = Int(input.dim(3))  # Input width
    let C = Int(input.dim(1))  # Number of input channels

    let R = Int(kernel.dim(2))  # Filter height
    let S = Int(kernel.dim(3))  # Filter width
    let F = Int(kernel.dim(1))  # Number of output channels

    let HO = (
        strides[0] * (H - 1)
        + output_padding[0]
        + ((R - 1) * dilations[0] + 1)
        - pads[PADS_H_START]
        - pads[PADS_H_END]
    ).to_int()
    let WO = (
        strides[1] * (W - 1)
        + output_padding[1]
        + ((S - 1) * dilations[1] + 1)
        - pads[PADS_W_START]
        - pads[PADS_W_END]
    ).to_int()

    for n in range(N):
        for c in range(C):
            for f in range(F):
                for i in range(H):
                    let indX_out = i * strides[0] - pads[PADS_H_START]
                    for j in range(W):
                        let indY_out = j * strides[1] - pads[PADS_W_START]
                        for r in range(R):
                            for s in range(S):
                                let x_out = (
                                    indX_out + r * dilations[0]
                                ).to_int()
                                let y_out = (
                                    indY_out + s * dilations[1]
                                ).to_int()
                                if (
                                    x_out >= 0
                                    and x_out < HO
                                    and y_out >= 0
                                    and y_out < WO
                                ):
                                    let tmp = output[n, f, x_out, y_out]
                                    output[
                                        StaticIntTuple[rank](n, f, x_out, y_out)
                                    ] = (
                                        tmp
                                        + input[n, c, i, j] * kernel[c, f, r, s]
                                    )

    # Add bias.
    for n in range(N):
        for i in range(HO):
            for j in range(WO):
                for f in range(F):
                    let index = f
                    output[StaticIntTuple[rank](n, f, i, j)] += epilogue_fn(
                        f, output[StaticIntTuple[rank](n, f, i, j)]
                    )

    out_chain.mark_ready()
