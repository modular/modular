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

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys import CompilationTarget
from std.testing import assert_equal, TestSuite

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


# Fields differ from `_HostBuffer`'s, so `_is_implicitly_encodable_to` rejects
# it; only the acceptor lets a `_HostBuffer` fill it.
struct _DeviceView(DeviceArgAcceptor, TrivialRegisterPassable):
    var lo: UInt32
    var hi: UInt32

    @staticmethod
    def _accepts_device_arg[ActualT: DevicePassable]() -> Bool:
        return ActualT == _HostBuffer

    @staticmethod
    def _device_arg_slot[
        ActualT: DevicePassable, target: CompilationTarget
    ]() -> SlotLayout:
        return SlotLayout(8, 8)


@fieldwise_init
struct _HostPair(DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self

    var buffer: _HostBuffer
    var count: Int64

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "_HostPair"


struct _DevicePair(TrivialRegisterPassable):
    var view: _DeviceView
    var count: Int64


def test_launcher_validates_accepted_arguments() raises:
    comptime assert not _HostPair._is_implicitly_encodable_to[_DevicePair]()
    var validated = _validate_encodable_arguments[
        CompilationTarget(),
        DeclaredArgTypes=TypeList.of[Trait=AnyType, _DeviceView, _DevicePair](),
        ActualArgTypes=TypeList.of[
            Trait=DevicePassable, _HostBuffer, _HostPair
        ](),
    ]()
    assert_equal(validated[0], 2)
    assert_equal(validated[1][0], 0)
    assert_equal(validated[1][1], 8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
