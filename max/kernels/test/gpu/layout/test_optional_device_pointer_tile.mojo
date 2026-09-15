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

"""`Optional` and `OptionalReg` kernel arguments convert their payload.

A `TileTensor` backed by `DevicePointerEngine` holds a host `DevicePointer`
that must be rewritten to a device address at the kernel boundary. Wrapping
the tile in `Optional` or `OptionalReg`, as a direct argument or as a closure
capture, must still run that conversion. Before GPUA-218 both optionals
bit-copied their payload, so the kernel read the host handle as a device
address and its stores went nowhere. On Metal the buffer was also never
registered with the encoder.

Each kernel writes a ramp through the tile when the optional is engaged and
must leave the buffer untouched when it is `None`.

`Optional[T].device_type` is `Optional[T.device_type]`, so an
`Optional[DeviceBuffer]` argument is declared by the kernel as an
`Optional[UnsafePointer]`; the last test covers that conversion.
"""

from layout import TensorLayout, TileTensor, row_major
from layout.tensor_engine import DevicePointerEngine
from std.collections import OptionalReg

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from std.testing import assert_equal

comptime _N = 8

# Origin-erased so the host-constructed tile's type matches the kernel
# parameter after `as_unsafe_any_origin()`.
comptime _Tile[LayoutType: TensorLayout] = TileTensor[
    .float32,
    LayoutType,
    UnsafeAnyOrigin[mut=True],
    Engine=DevicePointerEngine[element_width=1],
]


def _fill_ramp[LayoutType: TensorLayout](tile: _Tile[LayoutType]):
    for i in range(Int(tile.dim[0]())):
        tile.raw_store[width=1](i, Float32(i + 1))


def optional_reg_kernel[
    LayoutType: TensorLayout
](tile: OptionalReg[_Tile[LayoutType]]):
    if global_idx.x != 0:
        return
    if tile:
        _fill_ramp(tile.value())


def optional_kernel[
    LayoutType: TensorLayout
](tile: Optional[_Tile[LayoutType]]):
    if global_idx.x != 0:
        return
    if tile:
        _fill_ramp(tile.value())


def _check(
    ctx: DeviceContext, buf: DeviceBuffer[.float32], engaged: Bool
) raises:
    var host = ctx.enqueue_create_host_buffer[.float32](_N)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    for i in range(_N):
        assert_equal(host[i], Float32(i + 1) if engaged else Float32(-1))


def test_optional_reg_argument(ctx: DeviceContext, engaged: Bool) raises:
    print("== test_optional_reg_argument engaged =", engaged)
    var buf = ctx.enqueue_create_buffer[.float32](_N)
    buf.enqueue_fill(Float32(-1))
    var n = _N
    var tile = TileTensor(buf.device_ptr(), row_major(n))
    comptime Arg = OptionalReg[_Tile[tile.LayoutType]]
    var arg = Arg(tile.as_unsafe_any_origin()) if engaged else Arg()
    ctx.enqueue_function[optional_reg_kernel[tile.LayoutType]](
        arg, grid_dim=1, block_dim=1
    )
    _check(ctx, buf, engaged)


def test_optional_argument(ctx: DeviceContext, engaged: Bool) raises:
    print("== test_optional_argument engaged =", engaged)
    var buf = ctx.enqueue_create_buffer[.float32](_N)
    buf.enqueue_fill(Float32(-1))
    var n = _N
    var tile = TileTensor(buf.device_ptr(), row_major(n))
    comptime Arg = Optional[_Tile[tile.LayoutType]]
    var arg = Arg(tile.as_unsafe_any_origin()) if engaged else Arg()
    ctx.enqueue_function[optional_kernel[tile.LayoutType]](
        arg, grid_dim=1, block_dim=1
    )
    _check(ctx, buf, engaged)


def test_optional_reg_capture(ctx: DeviceContext, engaged: Bool) raises:
    print("== test_optional_reg_capture engaged =", engaged)
    var buf = ctx.enqueue_create_buffer[.float32](_N)
    buf.enqueue_fill(Float32(-1))
    var n = _N
    var tile = TileTensor(buf.device_ptr(), row_major(n))
    comptime Arg = OptionalReg[_Tile[tile.LayoutType]]
    var arg = Arg(tile.as_unsafe_any_origin()) if engaged else Arg()

    # The capture is encoded through `encode_closure_state`, which dispatches
    # to `OptionalReg._to_device_type` like a direct argument.
    def kernel() {var arg}:
        if arg:
            _fill_ramp(arg.value())

    ctx.enqueue_function(kernel, grid_dim=1, block_dim=1)
    _check(ctx, buf, engaged)


def optional_pointer_kernel(
    ptr: Optional[UnsafePointer[Float32, MutAnyOrigin]], n: Int32
):
    if global_idx.x != 0:
        return
    if ptr:
        var p = ptr.value()
        for i in range(Int(n)):
            p[unsafe_offset=i] = Float32(i + 1)


def test_optional_device_buffer_argument(
    ctx: DeviceContext, engaged: Bool
) raises:
    print("== test_optional_device_buffer_argument engaged =", engaged)
    var buf = ctx.enqueue_create_buffer[.float32](_N)
    buf.enqueue_fill(Float32(-1))
    comptime Arg = Optional[DeviceBuffer[.float32]]
    var arg = Arg(buf) if engaged else Arg()
    ctx.enqueue_function[optional_pointer_kernel](
        arg, Int32(_N), grid_dim=1, block_dim=1
    )
    _check(ctx, buf, engaged)


def main() raises:
    with DeviceContext() as ctx:
        test_optional_reg_argument(ctx, engaged=True)
        test_optional_reg_argument(ctx, engaged=False)
        test_optional_argument(ctx, engaged=True)
        test_optional_argument(ctx, engaged=False)
        test_optional_reg_capture(ctx, engaged=True)
        test_optional_reg_capture(ctx, engaged=False)
        test_optional_device_buffer_argument(ctx, engaged=True)
        test_optional_device_buffer_argument(ctx, engaged=False)
