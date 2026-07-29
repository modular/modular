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
"""CUDA native-handle accessors for the HAL `DeviceContext`."""

from std.ffi import _CPointer
from std.gpu.host import DeviceContext, DeviceFunction, DeviceStream
from std.memory.unsafe_pointer import unsafe_cast


struct _CUctx_st:
    pass


struct _CUstream_st:
    pass


struct _CUmod_st:
    pass


struct _CUevent_st:
    pass


comptime CUcontext = _CPointer[_CUctx_st, UntrackedOrigin[mut=True]]
comptime CUstream = _CPointer[_CUstream_st, UntrackedOrigin[mut=True]]
comptime CUmodule = _CPointer[_CUmod_st, UntrackedOrigin[mut=True]]
comptime CUevent = _CPointer[_CUevent_st, UntrackedOrigin[mut=True]]


@always_inline
def CUDA(ctx: DeviceContext) raises -> CUcontext:
    raise Error(
        "CUcontext access is not wired for the HAL DeviceContext; the plugin"
        " exposes it via M_driver_context_get_current_driver_context"
    )


@always_inline
def CUDA(stream: DeviceStream) raises -> CUstream:
    """Returns the `CUstream` backing a HAL stream.

    The HAL queue reports its backend stream through the `"native_handle"`
    property, which the CUDA plugin answers with the raw `CUstream`
    (`Target/lib/CUDA/M_driver_queue.cpp`).

    Args:
        stream: The HAL-backed stream to read the native handle from.

    Returns:
        The underlying `CUstream`.

    Raises:
        If the queue backing `stream` reports no native handle, which is the
        case for a device with no OS-level stream object.
    """
    return unsafe_cast[Type=_CUstream_st, origin=UntrackedOrigin[mut=True]](
        stream._native_stream()
    )


@always_inline
def CUDA_MODULE(func: DeviceFunction) raises -> CUmodule:
    raise Error(
        "CUmodule access is not wired for the HAL DeviceContext; the plugin"
        " exposes it via M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE"
    )


def CUDA_get_current_context() raises -> CUcontext:
    raise Error(
        "current-CUcontext access is not wired for the HAL DeviceContext; the"
        " plugin exposes it via M_driver_context_get_current_driver_context"
    )
