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

from std.testing import assert_equal

from max.gpu.host import DeviceContext
from layout import row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from matmul_kernels import gemm_kernel_6, matmul_kernel_tc
from max.gpu import WARP_SIZE


def a_value(row: Int, k: Int) -> Float32:
    return Float32((row + 2 * k) % 7 - 3)


def b_value(k: Int, col: Int) -> Float32:
    return Float32((3 * k + col) % 5 - 2)


def c_value(row: Int, col: Int) -> Float32:
    return Float32((row + col) % 9 - 4)


def test_views[
    M: Int, N: Int, K: Int, use_tensor_cores: Bool = False
](ctx: DeviceContext) raises:
    print("matmul views:", M, N, K, use_tensor_cores)
    comptime a_layout = row_major[M, K]()
    comptime b_layout = row_major[K, N]()
    comptime c_layout = row_major[M, N]()
    var a = HostDeviceTileTensor[.float32](a_layout, ctx)
    var b = HostDeviceTileTensor[.float32](b_layout, ctx)
    var c = HostDeviceTileTensor[.float32](c_layout, ctx)

    var a_host = a.host_tensor()
    var b_host = b.host_tensor()
    var c_host = c.host_tensor()
    comptime assert (
        a_host.flat_rank == b_host.flat_rank == c_host.flat_rank == 2
    )
    for row in range(M):
        for k in range(K):
            a_host[row, k] = a_value(row, k)
    for k in range(K):
        for col in range(N):
            b_host[k, col] = b_value(k, col)
    for row in range(M):
        for col in range(N):
            c_host[row, col] = 0 if use_tensor_cores else c_value(row, col)
    a.to_device()
    b.to_device()
    c.to_device()

    var a_device = a.device_tensor().as_imm().as_unsafe_any_origin()
    var b_device = b.device_tensor().as_imm().as_unsafe_any_origin()
    var c_device = c.device_tensor().as_unsafe_any_origin()
    comptime if use_tensor_cores:
        comptime MMA_N = 8 if ctx.target.is_nvidia_gpu() else 16
        comptime MMA_K = 8 if ctx.target.is_nvidia_gpu() else 4
        comptime kernel = matmul_kernel_tc[
            .float32,
            type_of(a_device).LayoutType,
            type_of(b_device).LayoutType,
            type_of(c_device).LayoutType,
            64,
            64,
            32,
            32,
            32,
            16,
            MMA_N,
            MMA_K,
        ]
        ctx.enqueue_function[kernel](
            a_device,
            b_device,
            c_device,
            grid_dim=(N // 64, M // 64),
            block_dim=(4 * WARP_SIZE,),
        )
    else:
        comptime kernel = gemm_kernel_6[
            .float32,
            type_of(a_device).LayoutType,
            type_of(b_device).LayoutType,
            type_of(c_device).LayoutType,
            128,
            128,
            8,
            8,
            8,
            256,
        ]
        ctx.enqueue_function[kernel](
            a_device,
            b_device,
            c_device,
            grid_dim=(N // 128, M // 128),
            block_dim=(256,),
        )
    c.to_host()

    # Integer-valued operands keep this layout oracle bit-exact under FMA.
    c_host = c.host_tensor()
    for row in range(M):
        for col in range(N):
            var expected = Float32(0) if use_tensor_cores else c_value(row, col)
            for k in range(K):
                expected += a_value(row, k) * b_value(k, col)
            assert_equal(c_host[row, col], expected)


def main() raises:
    with DeviceContext() as ctx:
        test_views[256, 128, 16](ctx)
        test_views[128, 256, 24](ctx)
        test_views[128, 64, 64, True](ctx)
        test_views[64, 128, 96, True](ctx)
