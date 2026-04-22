# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.gpu.host import DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from nn.conv.conv import conv_miopen

from std.utils.index import IndexList


def conv_ref_cpu[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input_ptr: UnsafePointer[Scalar[input_type], _],
    filter_ptr: UnsafePointer[Scalar[filter_type], _],
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    C_per_group: Int,
    F_out: Int,
    H_out: Int,
    W_out: Int,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad: IndexList[2],
    num_groups: Int,
):
    """CPU reference conv2d: input NHWC, filter RSCF, output NHWC."""
    var F_per_group = F_out // num_groups
    for n in range(N):
        for ho in range(H_out):
            for wo in range(W_out):
                for f in range(F_out):
                    var g = f // F_per_group
                    var ci_base = g * C_per_group
                    var acc = Float32(0)
                    for r in range(R):
                        for s in range(S):
                            var hi = ho * stride[0] - pad[0] + r * dilation[0]
                            var wi = wo * stride[1] - pad[1] + s * dilation[1]
                            if hi >= 0 and hi < H and wi >= 0 and wi < W:
                                for c in range(C_per_group):
                                    # NHWC: idx = n*H*W*C + h*W*C + w*C + c
                                    var in_idx = (
                                        n * H * W * C_in
                                        + hi * W * C_in
                                        + wi * C_in
                                        + ci_base
                                        + c
                                    )
                                    # RSCF: idx = r*S*C*F + s*C*F + c*F + f
                                    var f_idx = (
                                        r * S * C_per_group * F_out
                                        + s * C_per_group * F_out
                                        + c * F_out
                                        + f
                                    )
                                    acc += Float32(input_ptr[in_idx]) * Float32(
                                        filter_ptr[f_idx]
                                    )
                    # NHWC output
                    var out_idx = (
                        n * H_out * W_out * F_out
                        + ho * W_out * F_out
                        + wo * F_out
                        + f
                    )
                    output_ptr[out_idx] = Scalar[output_type](acc)


# input: NHWC
# filter: RSCF
def test_conv_miopen[
    input_dim: IndexList[4],
    filter_dim: IndexList[4],
    output_dim: IndexList[4],
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    stride_dim: IndexList[2],
    dilation_dim: IndexList[2],
    pad_dim: IndexList[
        4
    ],  # Format: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print(
        "== test_miopen_conv: dtype_in=",
        input_type,
        " dtype_filter=",
        filter_type,
        " dtype_out=",
        output_type,
        " input=",
        input_dim,
        " filter=",
        filter_dim,
        " num_groups=",
        num_groups,
    )

    # Extract dimensions
    comptime N = input_dim[0]
    comptime H = input_dim[1]
    comptime W = input_dim[2]
    comptime C_in = input_dim[3]

    comptime R = filter_dim[0]
    comptime S = filter_dim[1]
    comptime C = filter_dim[2]
    comptime F = filter_dim[3]

    comptime Nout = output_dim[0]
    comptime Hout = output_dim[1]
    comptime Wout = output_dim[2]
    comptime Cout = output_dim[3]

    comptime input_dim_flattened = N * H * W * C_in
    comptime filter_dim_flattened = R * S * C * F
    comptime output_dim_flattened = Nout * Hout * Wout * Cout

    # Allocate host memory
    var input_host_ptr = alloc[Scalar[input_type]](input_dim_flattened)
    var filter_host_ptr = alloc[Scalar[filter_type]](filter_dim_flattened)
    var output_ref_host_ptr = alloc[Scalar[output_type]](output_dim_flattened)
    var output_host_ptr = alloc[Scalar[output_type]](output_dim_flattened)

    # Create host TileTensors
    comptime input_tt_layout = row_major(
        (Idx[N](), Idx[H](), Idx[W](), Idx[C_in]())
    )
    comptime filter_tt_layout = row_major(
        (Idx[R](), Idx[S](), Idx[C](), Idx[F]())
    )
    comptime output_tt_layout = row_major(
        (Idx[Nout](), Idx[Hout](), Idx[Wout](), Idx[Cout]())
    )
    var input_host = TileTensor(input_host_ptr, input_tt_layout)
    var filter_host = TileTensor(filter_host_ptr, filter_tt_layout)
    var output_ref_host = TileTensor(output_ref_host_ptr, output_tt_layout)
    var output_host = TileTensor(output_host_ptr, output_tt_layout)

    random(input_host)
    random(filter_host)

    _ = output_host.fill(0)
    _ = output_ref_host.fill(0)

    # CPU reference
    var sym_pad = IndexList[2](pad_dim[0], pad_dim[2])
    conv_ref_cpu[input_type, filter_type, output_type](
        input_host_ptr,
        filter_host_ptr,
        output_ref_host_ptr,
        N,
        H,
        W,
        C_in,
        R,
        S,
        C,
        F,
        Hout,
        Wout,
        stride_dim,
        dilation_dim,
        sym_pad,
        num_groups,
    )

    # Allocate device buffers
    var input_dev = ctx.enqueue_create_buffer[input_type](input_dim_flattened)
    var filter_dev = ctx.enqueue_create_buffer[filter_type](
        filter_dim_flattened
    )
    var output_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )

    # Create device TileTensors
    var input_dev_tensor = TileTensor(input_dev, input_tt_layout)
    var filter_dev_tensor = TileTensor(filter_dev, filter_tt_layout)
    var output_dev_tensor = TileTensor(output_dev, output_tt_layout)

    ctx.enqueue_copy(input_dev, input_host_ptr)
    ctx.enqueue_copy(filter_dev, filter_host_ptr)

    # Test: MIOpen conv with RSCF filter layout
    conv_miopen[input_type, filter_type, output_type](
        input_dev_tensor,
        filter_dev_tensor,
        output_dev_tensor,
        stride_dim,
        dilation_dim,
        sym_pad,
        num_groups,
        ctx,
    )

    ctx.enqueue_copy(output_host_ptr, output_dev)
    ctx.synchronize()

    # Verify MIOpen output against CPU reference
    var max_diff = Float32(0)
    for i in range(output_dim_flattened):
        var diff = abs(
            Float32(output_host_ptr[i]) - Float32(output_ref_host_ptr[i])
        )
        if diff > max_diff:
            max_diff = diff
    # Use absolute tolerance appropriate for reduced precision types
    comptime if input_type == DType.float32:
        if max_diff > 1e-3:
            print("  FAIL: max_diff=", max_diff)
            raise Error("MIOpen output does not match CPU reference")
    else:
        # BF16/F16: accumulation over R*S*C elements with ~1e-2 relative
        # error per product. Observed max_diff ~0.25 for small 3x3x16
        # cases (BF16). 0.3 gives headroom without being overly loose.
        if max_diff > 0.3:
            print("  FAIL: max_diff=", max_diff)
            raise Error("MIOpen output does not match CPU reference")
    print("  PASS: max_diff=", max_diff)

    # Cleanup host memory
    input_host_ptr.free()
    filter_host_ptr.free()
    output_ref_host_ptr.free()
    output_host_ptr.free()

    # Cleanup device buffers
    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Test different data types
        comptime dtype_configs = (DType.float32, DType.float16, DType.bfloat16)

        comptime for i in range(len(dtype_configs)):
            comptime dtype = dtype_configs[i]

            test_conv_miopen[
                IndexList[4](1, 8, 8, 16),  # input  (NHWC)
                IndexList[4](3, 3, 16, 32),  # filter (RSCF)
                IndexList[4](1, 6, 6, 32),  # output (NHWC)
                dtype,
                dtype,
                dtype,
                IndexList[2](1, 1),  # stride
                IndexList[2](1, 1),  # dilation
                IndexList[4](
                    0, 0, 0, 0
                ),  # pad: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
            ](ctx)

        # Test with padding (Flux2 VAE decoder-style shape)
        test_conv_miopen[
            IndexList[4](1, 16, 16, 32),  # input  (NHWC)
            IndexList[4](3, 3, 32, 64),  # filter (RSCF)
            IndexList[4](1, 16, 16, 64),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](
                1, 1, 1, 1
            ),  # pad: [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
        ](ctx)

        # Test with stride=2
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 7, 7, 32),  # output (NHWC)
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            IndexList[2](2, 2),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
        ](ctx)

        # Test 1x1 convolution (channel projection)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 32),  # input  (NHWC)
            IndexList[4](1, 1, 32, 64),  # filter (RSCF)
            IndexList[4](1, 8, 8, 64),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
        ](ctx)

        # Test grouped convolution (2 groups)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 32),  # input  (NHWC)
            IndexList[4](3, 3, 16, 64),  # filter (RSCF) C_per_group=16
            IndexList[4](1, 6, 6, 64),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
            num_groups=2,
        ](ctx)

        # Test depthwise convolution (groups == C_in)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 16),  # input  (NHWC)
            IndexList[4](3, 3, 1, 16),  # filter (RSCF) C_per_group=1
            IndexList[4](1, 6, 6, 16),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
            num_groups=16,
        ](ctx)

        # Test dilation > 1
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 12, 12, 32),  # output (NHWC) H_out=(16-2*2)/1+1=12
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](2, 2),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
        ](ctx)

        # Test batch size > 1
        test_conv_miopen[
            IndexList[4](4, 8, 8, 16),  # input  (NHWC) N=4
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](4, 6, 6, 32),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
        ](ctx)

        # Test non-square spatial dims
        test_conv_miopen[
            IndexList[4](1, 12, 20, 16),  # input  (NHWC) H!=W
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 10, 18, 32),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
        ](ctx)

        # Test channel reduction (C_in > F_out)
        test_conv_miopen[
            IndexList[4](1, 8, 8, 64),  # input  (NHWC) C_in=64
            IndexList[4](3, 3, 64, 16),  # filter (RSCF) F_out=16
            IndexList[4](1, 6, 6, 16),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
        ](ctx)

        # Test 5x5 filter
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](5, 5, 16, 32),  # filter (RSCF)
            IndexList[4](1, 12, 12, 32),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](0, 0, 0, 0),  # no pad
        ](ctx)

        # Test stride + padding combined
        test_conv_miopen[
            IndexList[4](1, 16, 16, 16),  # input  (NHWC)
            IndexList[4](3, 3, 16, 32),  # filter (RSCF)
            IndexList[4](1, 8, 8, 32),  # output (NHWC) H_out=(16+2-3)/2+1=8
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](2, 2),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](1, 1, 1, 1),  # pad
        ](ctx)

        # Test larger dims (different MIOpen algorithm paths)
        test_conv_miopen[
            IndexList[4](2, 32, 32, 128),  # input  (NHWC)
            IndexList[4](3, 3, 128, 256),  # filter (RSCF)
            IndexList[4](2, 32, 32, 256),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[4](1, 1, 1, 1),  # pad
        ](ctx)

    print("\nAll MIOpen conv tests passed!")
