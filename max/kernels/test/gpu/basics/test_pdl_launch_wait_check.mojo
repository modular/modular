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
"""Tests the launch-time check that a PDL launch targets a kernel that waits.

Built with `-D MAX_CHECK_PDL_WAIT=True`, `DeviceFunction` compiles each kernel
to PTX at its launch and rejects a launch carrying the PDL attribute when the
kernel never executes `griddepcontrol.wait`. NVIDIA SM90+ only.
"""

from max.gpu.host import (
    DeviceContext,
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.sys import get_defined_bool
from std.testing import assert_equal, assert_true


def waiting_kernel(dst: MutPointer[Int32, MutAnyOrigin]):
    wait_on_dependent_grids()
    dst[0] = 1


def non_waiting_kernel(dst: MutPointer[Int32, MutAnyOrigin]):
    dst[0] = 1


def level_gated_kernel[
    pdl_level: PDLLevel
](dst: MutPointer[Int32, MutAnyOrigin]):
    # The shape of the kernels this check exists for: the wait is gated on a
    # `pdl_level` parameter, instantiated here with the default level.
    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()
    dst[0] = 1


def _launch_rejected[
    kernel: def(MutPointer[Int32, MutAnyOrigin]) thin -> None
](ctx: DeviceContext, dst: MutPointer[Int32, MutAnyOrigin]) raises -> Bool:
    try:
        ctx.enqueue_function[kernel](
            dst,
            grid_dim=1,
            block_dim=1,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    except e:
        assert_true("griddepcontrol.wait" in String(e), String(e))
        return True
    return False


def test_pdl_launch_wait_check() raises:
    comptime assert get_defined_bool[
        "MAX_CHECK_PDL_WAIT"
    ](), "build with -D MAX_CHECK_PDL_WAIT=True"
    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[DType.int32](1)
        var dst = buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

        # A kernel that waits launches normally with the PDL attribute.
        assert_true(not _launch_rejected[waiting_kernel](ctx, dst))

        # Without the PDL attribute nothing is checked.
        ctx.enqueue_function[non_waiting_kernel](
            dst,
            grid_dim=1,
            block_dim=1,
            attributes=pdl_launch_attributes(PDLLevel.OFF),
        )

        # The attribute carrying `False` disables PDL for the launch, so a
        # kernel with no wait is not rejected.
        ctx.enqueue_function[non_waiting_kernel](
            dst,
            grid_dim=1,
            block_dim=1,
            attributes=[
                LaunchAttribute(
                    LaunchAttributeID.PROGRAMMATIC_STREAM_SERIALIZATION,
                    LaunchAttributeValue(False),
                )
            ],
        )

        # With it, a kernel with no wait is rejected.
        assert_true(_launch_rejected[non_waiting_kernel](ctx, dst))

        # The bug class: the launch passes `pdl_level` to the attribute, but
        # the kernel was instantiated without it, so its wait compiled out.
        assert_true(_launch_rejected[level_gated_kernel[PDLLevel()]](ctx, dst))
        assert_true(
            not _launch_rejected[level_gated_kernel[PDLLevel.ON]](ctx, dst)
        )

        ctx.synchronize()
        var host = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(host, buf)
        ctx.synchronize()
        assert_equal(host[0], 1)


def main() raises:
    test_pdl_launch_wait_check()
