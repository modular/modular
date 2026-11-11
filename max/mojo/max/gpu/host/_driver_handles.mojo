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
"""Opaque AsyncRT handle types, and the nullable pointers to them.

The legacy and the HAL `DeviceContext` call the same C entry points, so both name
the same handle types and share these definitions.
"""

from std.ffi import CStringSpan
from std.memory import Pointer


# Create empty structs to ensure dtype checking when using the C++ handles.
struct _DeviceContextCpp:
    pass


struct _DeviceBufferCpp:
    pass


struct _DeviceFunctionCpp:
    pass


struct _DeviceMulticastBufferCpp:
    pass


struct _DeviceStreamCpp:
    pass


struct _DeviceEventCpp:
    pass


struct _DeviceTimerCpp:
    pass


struct _CompletionFlagCpp:
    pass


struct _DeviceContextScopeCpp:
    pass


struct _DeviceGraphBuilderCpp:
    pass


struct _DeviceGraphCpp:
    pass


comptime _DeviceContextPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceContextCpp, origin]]

comptime _DeviceBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceBufferCpp, origin]]

comptime _DeviceFunctionPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceFunctionCpp, origin]]

comptime _DeviceMulticastBufferPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceMulticastBufferCpp, origin]]

comptime _DeviceStreamPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceStreamCpp, origin]]

comptime _DeviceEventPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceEventCpp, origin]]

comptime _DeviceTimerPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceTimerCpp, origin]]

comptime _CompletionFlagPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_CompletionFlagCpp, origin]]

comptime _DeviceContextScopePtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceContextScopeCpp, origin]]

comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceGraphBuilderCpp, origin]]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = Optional[Pointer[_DeviceGraphCpp, origin]]

comptime _CString[origin: ImmOrigin = ImmUntrackedOrigin] = Optional[
    CStringSpan[origin]
]
