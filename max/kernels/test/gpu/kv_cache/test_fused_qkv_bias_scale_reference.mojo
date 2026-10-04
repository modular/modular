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
"""Checks public biased and scaled ragged QKV launches against scalar references."""

from std.collections import OptionalReg
from std.testing import assert_almost_equal, assert_equal
from std.utils import IndexList
from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    TileTensor,
    RowMajorLayout,
    row_major,
)
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.kv_cache_ragged import (
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale,
)


# Independent scalar formulas keep the reference outside the matmul/cache APIs.
def _input_value(row: Int, k: Int) -> Float32:
    return Float32((row + k) % 5 - 2) * 0.25


def _weight_value(col: Int, k: Int) -> Float32:
    return Float32((col * 3 + k) % 7 - 3) * 0.25


def _bias_value(col: Int) -> Float32:
    return Float32(col % 11 - 5) * 0.125


def _reference[
    input_dtype: DType, output_dtype: DType, mode: Int, with_bias: Bool
](row: Int, col: Int) -> Scalar[output_dtype]:
    var dot = Float32(0)
    for k in range(128):
        dot += (
            _input_value(row, k).cast[input_dtype]().cast[DType.float32]()
            * _weight_value(col, k).cast[input_dtype]().cast[DType.float32]()
        )
    comptime if mode == 1:
        dot *= 0.5 * 1.5
    elif mode == 2:
        dot *= Float32(row + 1) * 0.5 * Float32(col % 3 + 1)
    elif mode == 3:
        dot *= Float32(row + 1) * 0.5 * Float32(col // 128 + 1) * 2.0
    var result = dot.cast[output_dtype]()
    comptime if with_bias:
        result += _bias_value(col).cast[output_dtype]()
    return result


def _check_case[
    input_dtype: DType, output_dtype: DType, mode: Int, with_bias: Bool
](ctx: DeviceContext) raises:
    comptime H = 64
    comptime Q = 128
    comptime N = 256
    comptime K = 128
    comptime PAGE = 16
    comptime Collection = PagedKVCacheCollection[
        output_dtype,
        KVCacheStaticParams(num_heads=1, head_size=H),
        PAGE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var hidden = HostDeviceTileTensor[input_dtype](
        row_major(Int64(3), Idx[K]), ctx
    )
    var weight = HostDeviceTileTensor[input_dtype](
        row_major(Idx[N], Idx[K]), ctx
    )
    var output = HostDeviceTileTensor[output_dtype](
        row_major(Int64(3), Idx[Q]), ctx
    )
    var bias = HostDeviceTileTensor[output_dtype](row_major(Int64(N)), ctx)
    var offsets = HostDeviceTileTensor[DType.uint32](row_major(Int64(3)), ctx)
    var lengths = HostDeviceTileTensor[DType.uint32](row_major(Int64(2)), ctx)
    var lut = HostDeviceTileTensor[DType.uint32](
        row_major(Int64(2), Int64(2)), ctx
    )
    var blocks = HostDeviceTileTensor[output_dtype](
        row_major(Int64(5), Idx[2], Int64(2), Idx[PAGE], Idx[1], Idx[H]),
        ctx,
    )
    comptime SA0 = 1
    # The B200 blockwise path requires a 16-byte-aligned scale row stride.
    comptime SA1 = 4 if mode == 3 and output_dtype == .bfloat16 else (
        3 if mode == 2 or mode == 3 else 1
    )
    comptime SB0 = N if mode == 2 else 2 if mode == 3 else 1
    var scale_a = HostDeviceTileTensor[DType.float32](
        row_major(Int64(SA0), Int64(SA1)), ctx
    )
    var scale_b = HostDeviceTileTensor[DType.float32](
        row_major(Int64(SB0), Int64(1)), ctx
    )
    for row in range(3):
        for k in range(K):
            hidden.host_tensor()[row, k] = _input_value(row, k).cast[
                input_dtype
            ]()
    for col in range(N):
        bias.host_tensor()[col] = _bias_value(col).cast[output_dtype]()
        for k in range(K):
            weight.host_tensor()[col, k] = _weight_value(col, k).cast[
                input_dtype
            ]()
    for i in range(SA0):
        for j in range(SA1):
            scale_a.host_tensor()[i, j] = Float32(i + j + 1) * 0.5
    for i in range(SB0):
        comptime if mode == 2:
            scale_b.host_tensor()[i, 0] = Float32(i % 3 + 1)
        elif mode == 1:
            scale_b.host_tensor()[i, 0] = 1.5
        else:
            scale_b.host_tensor()[i, 0] = Float32(i + 1) * 2.0
    offsets.host_tensor()[0] = 0
    offsets.host_tensor()[1] = 2
    offsets.host_tensor()[2] = 3
    lengths.host_tensor()[0] = 15
    lengths.host_tensor()[1] = 3
    lut.host_tensor()[0, 0] = 3
    lut.host_tensor()[0, 1] = 1
    lut.host_tensor()[1, 0] = 2
    lut.host_tensor()[1, 1] = 0
    _ = output.host_tensor().fill(-999)
    _ = blocks.host_tensor().fill(-999)
    hidden.to_device()
    weight.to_device()
    bias.to_device()
    offsets.to_device()
    lengths.to_device()
    lut.to_device()
    output.to_device()
    blocks.to_device()
    scale_a.to_device()
    scale_b.to_device()
    var collection = Collection(
        rebind[Collection.blocks_tt_type](
            blocks.device_tensor().as_unsafe_any_origin()
        ),
        lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(2),
        UInt32(17),
    )
    # Exercise the public launchers rather than private epilogue leaves.
    comptime if mode == 0:
        comptime assert input_dtype == output_dtype and with_bias
        generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[target="gpu"](
            hidden.device_tensor().as_imm().as_unsafe_any_origin(),
            offsets.device_tensor().as_imm().as_unsafe_any_origin(),
            weight.device_tensor().as_imm().as_unsafe_any_origin(),
            collection,
            UInt32(1),
            output.device_tensor()
            .bitcast[input_dtype]()
            .as_unsafe_any_origin(),
            bias.device_tensor()
            .bitcast[input_dtype]()
            .as_imm()
            .as_unsafe_any_origin(),
            ctx,
        )
    else:
        comptime BiasType = TileTensor[
            output_dtype, RowMajorLayout[Int64], ImmutAnyOrigin
        ]
        var maybe_bias: OptionalReg[BiasType] = None
        comptime if with_bias:
            maybe_bias = rebind[BiasType](
                bias.device_tensor().as_imm().as_unsafe_any_origin()
            )
        comptime granularity = IndexList[3](
            -1, -1, -1
        ) if mode == 1 else IndexList[3](1, 1, -1) if mode == 2 else IndexList[
            3
        ](
            1, 128, 128
        )
        generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
            scales_granularity_mnk=granularity, target="gpu"
        ](
            hidden.device_tensor().as_imm().as_unsafe_any_origin(),
            offsets.device_tensor().as_imm().as_unsafe_any_origin(),
            weight.device_tensor().as_imm().as_unsafe_any_origin(),
            scale_a.device_tensor().as_imm().as_unsafe_any_origin(),
            scale_b.device_tensor().as_imm().as_unsafe_any_origin(),
            collection,
            UInt32(1),
            output.device_tensor().as_unsafe_any_origin(),
            ctx,
            maybe_bias,
        )
    output.to_host()
    blocks.to_host()
    for row in range(3):
        for col in range(Q):
            assert_almost_equal(
                output.host_tensor()[row, col],
                _reference[input_dtype, output_dtype, mode, with_bias](
                    row, col
                ),
                atol=0.01,
                rtol=0.01,
            )
    # Inspect the full physical allocation, including prefixes, the other layer,
    # unused pages and padding after the appended tokens.
    for page in range(5):
        for kv in range(2):
            for layer in range(2):
                for token in range(PAGE):
                    var row = -1
                    if layer == 1:
                        if page == 3 and token == 15:
                            row = 0
                        elif page == 1 and token == 0:
                            row = 1
                        elif page == 2 and token == 3:
                            row = 2
                    for hd in range(H):
                        var actual = blocks.host_tensor()[
                            page, kv, layer, token, 0, hd
                        ]
                        if row >= 0:
                            assert_almost_equal(
                                actual,
                                _reference[
                                    input_dtype, output_dtype, mode, with_bias
                                ](row, Q + kv * H + hd),
                                atol=0.01,
                                rtol=0.01,
                            )
                        else:
                            assert_equal(actual, Scalar[output_dtype](-999))


def main() raises:
    with DeviceContext() as ctx:
        _check_case[DType.float32, DType.float32, 0, True](ctx)
        _check_case[DType.bfloat16, DType.bfloat16, 0, True](ctx)
        comptime for mode in range(1, 4):
            _check_case[DType.float8_e4m3fn, DType.float32, mode, False](ctx)
            _check_case[DType.float8_e4m3fn, DType.float32, mode, True](ctx)
        # BF16 output selects the specialized blockwise kernel on B200.
        _check_case[DType.float8_e4m3fn, DType.bfloat16, 3, False](ctx)
        _check_case[DType.float8_e4m3fn, DType.bfloat16, 3, True](ctx)
