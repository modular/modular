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
"""HIP native-handle accessors for the HAL `DeviceContext`."""

from .device_context_hal import DeviceContext, DeviceFunction, DeviceStream
from std.memory.unsafe_pointer import unsafe_cast


struct _ihipDevice_t:
    pass


struct _ihipStream_t:
    pass


struct _ihipModule_t:
    pass


comptime hipDevice_t = Optional[
    Pointer[_ihipDevice_t, UntrackedOrigin[mut=True]]
]
comptime hipStream_t = Optional[
    Pointer[_ihipStream_t, UntrackedOrigin[mut=True]]
]
comptime hipModule_t = Optional[
    Pointer[_ihipModule_t, UntrackedOrigin[mut=True]]
]


@inline(.always)
def HIP(ctx: DeviceContext) raises -> hipDevice_t:
    raise Error("hipDevice_t access is not wired for the HAL DeviceContext")


@inline(.always)
def HIP(stream: DeviceStream) raises -> hipStream_t:
    """Returns the `hipStream_t` backing a HAL stream.

    Args:
        stream: The stream to read the handle from.

    Returns:
        The underlying `hipStream_t`.

    Raises:
        If the stream has no native handle.
    """
    return unsafe_cast[Type=_ihipStream_t, origin=UntrackedOrigin[mut=True]](
        stream._native_stream()
    )


@inline(.always)
def HIP_MODULE(func: DeviceFunction) raises -> hipModule_t:
    raise Error(
        "hipModule_t access is not wired for the HAL DeviceContext; the HIP"
        " plugin does not answer M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE"
    )
