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
    out_chain: OutputChainPtr,
):
    """
    Implements the ConvTranspose operator from the MO spec.

    Parameters:
        rank: Rank of the input, output, and kernel tensors.
        type: Type of the input, output, and kernel tensors.
        strides_type: Element type of strides,
        dilations_type: Element type of dilations,
        pads_type: Element type of pads,

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
        out_chain: The OutputChainPtr used to mark competion or error of the task.
    """

    let N = Int(input.dim(0))  # Number of images (num. batches)
    let H = Int(input.dim(2))  # Input height
    let W = Int(input.dim(3))  # Input width
    let C = Int(input.dim(1))  # Number of input channels

    let R = Int(kernel.dim(2))  # Filter height
    let S = Int(kernel.dim(3))  # Filter width
    let F = Int(kernel.dim(1))  # Number of output channels

    let HO = Int(output.dim(2))
    let WO = Int(output.dim(3))

    for n in range(N):
        for c in range(C):
            for f in range(F):
                for i in range(H):
                    let indX_out = i * strides[0].to_int() - pads[
                        PADS_H_START
                    ].to_int()
                    for j in range(W):
                        let indY_out = j * strides[1].to_int() - pads[
                            PADS_W_START
                        ].to_int()
                        for r in range(R):
                            for s in range(S):
                                let x_out = indX_out + r * dilations[0].to_int()
                                let y_out = indY_out + s * dilations[1].to_int()
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
    out_chain.mark_ready()
