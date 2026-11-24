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
"""Real by-value GPU launch test for `Coord`'s `DevicePassable` conformance.

`test_device_passable.mojo` covers `Coord`'s host-side encoding (bytes into a
host buffer via `DefaultDeviceTypeEncoder`). This test closes the remaining gap
by launching a kernel that receives a `Coord` and reads it back on the device,
so the host->device transfer is exercised end to end.

`Coord` reaches a launch three ways, and each is covered: embedded in a
`DevicePassable` composite (as `TileTensor.layout` carries `Coord`
`_shape`/`_stride`), as a bare kernel argument, and captured by a kernel
closure. The bare and captured cases use several dynamic leaves and write a
trailing value back, so a launch whose argument count disagrees with the
kernel's parameters reads a wrong value instead of passing.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host import DeviceContext
from std.testing import assert_equal, TestSuite
from std.utils.coord import Coord, CoordLike, Idx

comptime _TAG = 0x5A5A_1234


# A `DevicePassable` composite carrying a `Coord` field, encoding via
# `encode_fields` exactly as `TileTensor` does for its `Layout` (whose
# `_shape`/`_stride` are `Coord`s).
@fieldwise_init
struct CoordBox(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    var dims: Coord[Int64, Int64]
    var tag: Int64

    comptime device_type = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return String("CoordBox")


# Launched single-threaded (grid=1/block=1), so the kernel writes
# unconditionally and needs no `global_idx`.
def _read_back_kernel(box: CoordBox, dst: Pointer[Int64, MutAnyOrigin]):
    dst[] = Int64(Int(box.dims[0]))
    dst[unsafe_offset=1] = Int64(Int(box.dims[1]))
    dst[unsafe_offset=2] = Int64(box.tag)


def test_coord_field_survives_device_launch() raises:
    with DeviceContext() as ctx:
        var box = CoordBox(
            dims=Coord[Int64, Int64](Int64(11), Int64(22)), tag=33
        )
        var out_dev = ctx.enqueue_create_buffer[.int64](3)
        ctx.enqueue_function[_read_back_kernel](
            box, out_dev.unsafe_ptr(), grid_dim=1, block_dim=1
        )
        var out_host = ctx.enqueue_create_host_buffer[.int64](3)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        # The `Coord` field transferred intact through the launch boundary.
        assert_equal(Int(out_host[0]), 11)
        assert_equal(Int(out_host[1]), 22)
        assert_equal(Int(out_host[2]), 33)
        _ = out_dev^


def _read_back_coord_kernel[
    element_types: TypeList[Trait=CoordLike, ...]
](
    c: Coord[*element_types],
    tag: Int64,
    dst: MutPointer[Int64, MutUntrackedOrigin],
):
    comptime for i in range(c.rank):
        dst[unsafe_offset=i] = Int64(Int(c[i].value()))
    dst[unsafe_offset=c.rank] = tag


def _check_bare_coord_arg(c: Coord, expected: List[Int]) raises:
    comptime rank = type_of(c).rank
    assert_equal(len(expected), rank)
    with DeviceContext() as ctx:
        var out_dev = ctx.enqueue_create_buffer[.int64](rank + 1)
        out_dev.enqueue_fill(-1)
        ctx.enqueue_function[_read_back_coord_kernel[type_of(c).element_types]](
            c, Int64(_TAG), out_dev.unsafe_ptr(), grid_dim=1, block_dim=1
        )
        var out_host = ctx.enqueue_create_host_buffer[.int64](rank + 1)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        for i in range(rank):
            assert_equal(Int(out_host[i]), expected[i])
        assert_equal(Int(out_host[rank]), _TAG)
        _ = out_dev^


def test_bare_dynamic_coord_arg() raises:
    _check_bare_coord_arg(
        Coord(Int64(3), Int64(5), Int64(7), Int64(11), Int64(13)),
        [3, 5, 7, 11, 13],
    )


def test_bare_mixed_coord_arg() raises:
    # The `ComptimeInt` leaves take no argument space, leaving two dynamic
    # leaves in a struct whose type still lists four.
    _check_bare_coord_arg(
        Coord(Idx[8], Idx[16], Int64(7), Int64(11)), [8, 16, 7, 11]
    )


def test_captured_dynamic_coord() raises:
    comptime rank = 3
    var c = Coord(Int64(3), Int64(5), Int64(7))
    var tag = Int64(_TAG)
    with DeviceContext() as ctx:
        var out_dev = ctx.enqueue_create_buffer[.int64](rank + 1)
        out_dev.enqueue_fill(-1)
        var dst = out_dev.unsafe_ptr()

        def kernel() {var c, var tag, var dst}:
            comptime for i in range(rank):
                dst[unsafe_offset=i] = Int64(Int(c[i].value()))
            dst[unsafe_offset=rank] = tag

        ctx.enqueue_closure(kernel, grid_dim=1, block_dim=1)
        var out_host = ctx.enqueue_create_host_buffer[.int64](rank + 1)
        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()
        assert_equal(Int(out_host[0]), 3)
        assert_equal(Int(out_host[1]), 5)
        assert_equal(Int(out_host[2]), 7)
        assert_equal(Int(out_host[rank]), _TAG)
        _ = out_dev^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
