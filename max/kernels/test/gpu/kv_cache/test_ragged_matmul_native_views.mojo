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

"""Checks ragged matmul epilogues with dynamic rows and padded outputs."""

from std.sys import has_amd_gpu_accelerator
from std.testing import assert_equal
from std.utils.index import IndexList

from max.gpu.host import DeviceContext
from layout import Coord, Idx, MixedLayout, TileTensor, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.kv_cache_ragged import _matmul_common


def _project[
    dtype: DType, //, target: StaticString
](
    hidden: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    weight: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: Optional[DeviceContext],
) raises:
    @__parameter
    @__copy_capture(output)
    @inline(.always)
    def write_output[
        value_type: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], value: SIMD[value_type, width]):
        output.store[width=width](Coord(idx), value.cast[dtype]())

    _matmul_common[target=target, elementwise_lambda_fn=write_output](
        hidden, weight, context
    )


def _check_projection[
    dtype: DType, target: StaticString, input_padding: Int = 0
](ctx: DeviceContext,) raises:
    comptime M = 5
    comptime N = 64
    comptime K = 64
    comptime output_padding = 16
    var hidden = HostDeviceTileTensor[dtype](
        row_major[M, K + input_padding](), ctx
    )
    var weight = HostDeviceTileTensor[dtype](
        row_major[N, K + input_padding](), ctx
    )
    var output = HostDeviceTileTensor[dtype](
        row_major[M, N + output_padding](), ctx
    )
    var hidden_host = hidden.host_tensor().fill(111)
    var weight_host = weight.host_tensor().fill(222)
    var output_host = output.host_tensor().fill(-777)
    comptime assert hidden_host.flat_rank == 2
    comptime assert weight_host.flat_rank == 2
    comptime assert output_host.flat_rank == 2
    for row in range(M):
        for k in range(K):
            hidden_host[row, k] = Scalar[dtype](
                Float32((row + 3 * k) % 17 - 8) * 0.125
            )
    for col in range(N):
        for k in range(K):
            weight_host[col, k] = Scalar[dtype](
                Float32((3 * col + k) % 13 - 6) * 0.125
            )

    var rows = M
    var hidden_layout = MixedLayout(
        Coord(rows, Idx[K]), Coord(Idx[K + input_padding], Idx[1])
    )
    var weight_layout = MixedLayout(
        Coord(Idx[N], Idx[K]), Coord(Idx[K + input_padding], Idx[1])
    )
    comptime if target == "cpu":
        _project[target=target](
            TileTensor(hidden_host.as_imm().unsafe_ptr(), hidden_layout),
            TileTensor(weight_host.as_imm().unsafe_ptr(), weight_layout),
            output_host,
            None,
        )
    else:
        hidden.to_device()
        weight.to_device()
        output.to_device()
        _project[target=target](
            TileTensor(
                hidden.device_tensor().as_imm().unsafe_ptr(), hidden_layout
            ),
            TileTensor(
                weight.device_tensor().as_imm().unsafe_ptr(), weight_layout
            ),
            output.device_tensor(),
            ctx,
        )
        output.to_host()

    # Dyadic inputs make the FP32 dot products exact before the output cast.
    for row in range(M):
        for col in range(N):
            var expected = Float32(0)
            for k in range(K):
                expected += Float32(hidden_host[row, k]) * Float32(
                    weight_host[col, k]
                )
            assert_equal(
                output_host[row, col],
                expected.cast[dtype](),
                msg=String(target) + " " + String(dtype),
            )
        for col in range(N, N + output_padding):
            assert_equal(output_host[row, col], Scalar[dtype](-777))


def main() raises:
    with DeviceContext() as ctx:
        _check_projection[DType.float32, "cpu"](ctx)
        _check_projection[DType.float32, "gpu"](ctx)
        _check_projection[DType.bfloat16, "gpu"](ctx)
        # Some legacy dispatches require dense input rows; AMD also covers padding.
        comptime if has_amd_gpu_accelerator():
            _check_projection[DType.float32, "gpu", 16](ctx)
            _check_projection[DType.bfloat16, "gpu", 16](ctx)
