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
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, coord, row_major, stack_allocation
from layout.tile_layout import Layout
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tensor_core import TensorCore


@inline(.always)
def a_value(row: Int, k: Int) -> Float16:
    return Float16((row * 3 + k * 2) % 7 - 3)


@inline(.always)
def b_value(k: Int, col: Int) -> Float16:
    return Float16((k * 3 + col * 5) % 9 - 4)


@inline(.always)
def c_value(row: Int, col: Int) -> Float32:
    return Float32((row + 2 * col) % 11 - 5)


def bulk_tile_kernel[
    MG: Int, NG: Int, IM: Int, IN: Int, output_layout: TensorLayout
](
    output: TileTensor[.float32, output_layout, MutAnyOrigin],
    legacy_output: TileTensor[.float32, output_layout, MutAnyOrigin],
):
    comptime IK = 16
    comptime KG = 3
    var tc = TensorCore[.float32, .float16, (IM, IN, IK)]()
    comptime AR = type_of(tc).a_reg_type.length
    comptime BR = type_of(tc).b_reg_type.length
    comptime CR = type_of(tc).c_reg_type.length
    # Padding exercises scalar-aligned groups with non-vector-aligned strides.
    var a_storage = stack_allocation[.float16, address_space=.LOCAL](
        row_major[MG, AR + 1]()
    )
    var a = TileTensor(
        a_storage.unsafe_ptr(),
        Layout(coord[MG, AR], coord[AR + 1, 1]),
    )
    var b_storage = stack_allocation[.float16, address_space=.LOCAL](
        row_major[NG, BR + 1]()
    )
    var b = TileTensor(
        b_storage.unsafe_ptr(),
        Layout(coord[NG, BR], coord[BR + 1, 1]),
    )
    var c_storage = stack_allocation[.float32, address_space=.LOCAL](
        row_major[MG * NG, CR + 1]()
    )
    var c = TileTensor(
        c_storage.unsafe_ptr(),
        Layout(coord[MG * NG, CR], coord[CR + 1, 1]),
    )
    var legacy_c_storage = stack_allocation[.float32, address_space=.LOCAL](
        row_major[MG * NG, CR + 1]()
    )
    var legacy_c = TileTensor(
        legacy_c_storage.unsafe_ptr(),
        Layout(coord[MG * NG, CR], coord[CR + 1, 1]),
    )
    var a_tile = stack_allocation[.float16, address_space=.LOCAL](
        row_major[IM, IK]()
    )
    var b_tile = stack_allocation[.float16, address_space=.LOCAL](
        row_major[IK, IN]()
    )
    var c_tile = stack_allocation[.float32, address_space=.LOCAL](
        row_major[IM, IN]()
    )
    comptime for m in range(MG):
        comptime for n in range(NG):
            for r in range(IM):
                for col in range(IN):
                    c_tile[r, col] = c_value(m * IM + r, n * IN + col)
            var fragment = tc.load_c(c_tile)
            comptime for reg in range(CR):
                c[n * MG + m, reg] = fragment[0, reg]
                legacy_c[n * MG + m, reg] = fragment[0, reg]
    comptime for kg in range(KG):
        comptime for m in range(MG):
            for r in range(IM):
                for k in range(IK):
                    a_tile[r, k] = a_value(m * IM + r, kg * IK + k)
            var fragment = tc.load_a(a_tile)
            comptime for reg in range(AR):
                a[m, reg] = fragment[0, reg]
        comptime for n in range(NG):
            for k in range(IK):
                for col in range(IN):
                    b_tile[k, col] = b_value(kg * IK + k, n * IN + col)
            var fragment = tc.load_b(b_tile)
            comptime for reg in range(BR):
                b[n, reg] = fragment[reg, 0]
        tc.mma(a.vectorize[1, AR](), b.vectorize[1, BR](), c.vectorize[1, CR]())
        tc.mma(
            a.to_layout_tensor().vectorize[1, AR](),
            b.to_layout_tensor().vectorize[1, BR](),
            legacy_c.to_layout_tensor().vectorize[1, CR](),
        )
    comptime for m in range(MG):
        comptime for n in range(NG):
            var fragment = stack_allocation[.float32, address_space=.LOCAL](
                row_major[1, CR]()
            )
            comptime for reg in range(CR):
                fragment[0, reg] = c[n * MG + m, reg]
            tc.store_d(output.tile[IM, IN](m, n), fragment)
            comptime for reg in range(CR):
                fragment[0, reg] = legacy_c[n * MG + m, reg]
            tc.store_d(legacy_output.tile[IM, IN](m, n), fragment)


def test_bulk_tile[
    MG: Int, NG: Int, IM: Int, IN: Int
](ctx: DeviceContext) raises:
    comptime K = 48
    comptime output_layout = row_major[MG * IM, NG * IN]()
    var output = HostDeviceTileTensor[.float32](output_layout, ctx)
    var legacy = HostDeviceTileTensor[.float32](output_layout, ctx)
    _ = output.host_tensor().fill(-999)
    _ = legacy.host_tensor().fill(-999)
    output.to_device()
    legacy.to_device()
    ctx.enqueue_function[
        bulk_tile_kernel[MG, NG, IM, IN, type_of(output_layout)]
    ](
        output.device_tensor().as_unsafe_any_origin(),
        legacy.device_tensor().as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    output.to_host()
    legacy.to_host()
    var actual = output.host_tensor()
    var previous = legacy.host_tensor()
    for row in range(MG * IM):
        for col in range(NG * IN):
            var expected = c_value(row, col)
            for k in range(K):
                expected += (
                    a_value(row, k).cast[.float32]()
                    * b_value(k, col).cast[.float32]()
                )
            assert_equal(actual[row, col], expected)
            assert_equal(previous[row, col], expected)


def main() raises:
    var ctx = DeviceContext()
    comptime if ctx.target.is_nvidia_gpu():
        test_bulk_tile[2, 3, 16, 8](ctx)
        test_bulk_tile[3, 2, 16, 8](ctx)
    else:
        test_bulk_tile[2, 3, 32, 32](ctx)
        test_bulk_tile[3, 2, 32, 32](ctx)
