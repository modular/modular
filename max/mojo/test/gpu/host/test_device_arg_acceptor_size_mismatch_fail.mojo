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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys import CompilationTarget

from max.gpu.host._launch_args import (
    DeviceArgAcceptor,
    SlotLayout,
    _validate_encodable_arguments,
)


@fieldwise_init
struct _HostBuffer(DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self

    var addr: UInt64

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "_HostBuffer"


# Accepts a `_HostBuffer` but claims a slot wider than the 8 bytes it encodes.
struct _WideView(DeviceArgAcceptor, TrivialRegisterPassable):
    var lo: UInt64
    var hi: UInt64

    @staticmethod
    def _accepts_device_arg[ActualT: DevicePassable]() -> Bool:
        return ActualT == _HostBuffer

    @staticmethod
    def _device_arg_slot[
        ActualT: DevicePassable, target: CompilationTarget
    ]() -> SlotLayout:
        return SlotLayout(16, 8)


def main():
    # CHECK: argument #0 of type '_HostBuffer' encodes to 8 bytes but the declared function argument type '{{.*}}_WideView' is 16 bytes
    _ = _validate_encodable_arguments[
        CompilationTarget(),
        DeclaredArgTypes=TypeList.of[Trait=AnyType, _WideView](),
        ActualArgTypes=TypeList.of[Trait=DevicePassable, _HostBuffer](),
    ]()
