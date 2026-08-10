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

"""Regression sweep for static conv2d with wide width padding."""

from std.math import ceildiv
from std.sys.info import simd_width_of
from std.testing import assert_equal

from layout import IntTuple, Layout, LayoutTensor
from nn.conv.conv import ConvDirectNHWC, ConvInfoStatic, pack_filter_lt
from nn.conv.conv_utils import ConvShape, get_micro_kernel_shape
from std.utils.index import Index


comptime N = 1
comptime H = 1
comptime W = 16
comptime C = 3
comptime R = 1
comptime F = 2
comptime stride_h = 1
comptime dilation_h = 1
comptime value_type = DType.float32
comptime simd_size = simd_width_of[value_type]()


def run_case[
    K: Int,
    stride_w: Int,
    dilation_w: Int,
    pad_left: Int,
    pad_right: Int,
]() raises:
    comptime HO = 1
    comptime WO = (
        W + pad_left + pad_right - dilation_w * (K - 1) - 1
    ) // stride_w + 1
    comptime conv_attr = ConvInfoStatic[2](
        IntTuple(0, pad_left, 0, pad_right),
        IntTuple(stride_h, stride_w),
        IntTuple(dilation_h, dilation_w),
        1,
    )
    comptime micro_kernel_shape = get_micro_kernel_shape[
        2, WO, F, conv_attr, simd_size
    ]()
    comptime micro_kernel_f_size = micro_kernel_shape[1] * simd_size
    comptime num_micro_tiles = ceildiv(F, micro_kernel_f_size)

    var input_storage = InlineArray[Scalar[value_type], N * H * W * C](fill=1.0)
    var filter_storage = InlineArray[
        Scalar[value_type],
        num_micro_tiles * R * K * C * micro_kernel_f_size,
    ](fill=0.0)
    var unpacked_filter_storage = InlineArray[
        Scalar[value_type], R * K * C * F
    ](fill=1.0)
    var output_storage = InlineArray[Scalar[value_type], N * HO * WO * F](
        fill=0.0
    )

    var input = LayoutTensor[value_type, Layout.row_major(N, H, W, C)](
        input_storage
    )
    var filter = LayoutTensor[
        value_type,
        Layout.row_major(num_micro_tiles, R, K, C, micro_kernel_f_size),
    ](filter_storage)
    var unpacked_filter = LayoutTensor[
        value_type, Layout.row_major(R, K, C, F)
    ](unpacked_filter_storage)
    var output = LayoutTensor[value_type, Layout.row_major(N, HO, WO, F)](
        output_storage
    )
    var conv_shape = ConvShape[2](
        n=N,
        input_dims=Index(H, W),
        output_dims=Index(HO, WO),
        filter_dims=Index(R, K),
        c=C,
        f=F,
        stride=Index(stride_h, stride_w),
        dilation=Index(dilation_h, dilation_w),
        pad_d=Index(0, 0),
        pad_h=Index(0, 0),
        pad_w=Index(pad_left, pad_right),
        num_groups=1,
    )

    pack_filter_lt[simd_size, micro_kernel_f_size](unpacked_filter, filter, 1)

    ConvDirectNHWC[
        Layout.row_major(N, H, W, C),
        Layout.row_major(num_micro_tiles, R, K, C, micro_kernel_f_size),
        Layout.row_major(N, HO, WO, F),
        value_type,
        value_type,
        value_type,
        True,
        conv_attr,
    ].run(output, input, filter, conv_shape)

    for wo in range(WO):
        var valid_count = 0
        for s in range(K):
            var input_w = wo * stride_w + s * dilation_w - pad_left
            if 0 <= input_w < W:
                valid_count += 1
        var expected = Float32(valid_count * C)
        for f in range(F):
            var actual = output_storage[wo * F + f]
            assert_equal(
                actual,
                expected,
                msg=String(
                    t"K={K} stride={stride_w} dilation={dilation_w} pads={pad_left},{pad_right} wo={wo} f={f}"
                ),
            )


# CHECK: PASS
def main() raises:
    comptime for k in range(2, 13):
        run_case[k, 1, 1, k - 1, k - 1]()
    run_case[8, 2, 1, 14, 14]()
    run_case[8, 1, 2, 14, 14]()
    run_case[8, 2, 2, 14, 14]()
    run_case[8, 2, 1, 13, 9]()
    run_case[8, 2, 1, 9, 15]()
    print("PASS")
