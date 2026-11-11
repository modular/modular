# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Checks that the Mojo flag a graph op rebuilds from a C API flag's graph
# handle reads the value signaled through the C API.

from max.gpu.host import CompletionFlag, DeviceContext
from std.ffi import external_call
from std.testing import assert_equal


def _check(status: Int32, what: StaticString) raises:
    if status != 0:
        raise Error(t"{what} failed with status {status}")


def main() raises:
    with DeviceContext() as ctx:
        if not ctx.supports_flag_wait():
            print("Skipping: no flag waits on", ctx.api())
            return

    var device = 0
    _check(
        external_call["M_DriverAcceleratorDeviceCreate", Int32](
            Int64(0), Pointer(to=device)
        ),
        "M_DriverAcceleratorDeviceCreate",
    )
    var flag = 0
    _check(
        external_call["M_DriverCompletionFlagCreate", Int32](
            device, Pointer(to=flag)
        ),
        "M_DriverCompletionFlagCreate",
    )
    var handle: UInt64 = 0
    _check(
        external_call["M_DriverCompletionFlagGraphHandle", Int32](
            flag, Pointer(to=handle)
        ),
        "M_DriverCompletionFlagGraphHandle",
    )
    _check(
        external_call["M_DriverCompletionFlagSignal", Int32](flag, UInt64(7)),
        "M_DriverCompletionFlagSignal",
    )
    var view = CompletionFlag(unsafe_from_address=Int(handle))
    assert_equal(view.load(), 7)
    _ = view^
    _check(
        external_call["M_DriverCompletionFlagDestroy", Int32](flag),
        "M_DriverCompletionFlagDestroy",
    )
    _check(
        external_call["M_DriverDeviceDestroy", Int32](device),
        "M_DriverDeviceDestroy",
    )
