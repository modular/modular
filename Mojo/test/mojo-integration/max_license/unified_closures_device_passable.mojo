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
# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 %mojo %s 2 1234 5678
# RUN: %mojo %s 2 1234 5678

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu.host import DefaultDeviceTypeEncoder
from std.reflection import reflect
from std.testing import assert_equal
from std.sys import argv


# Host-side DevicePassable types with non-identity device representations.
@fieldwise_init
struct HostInt(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = Int
    var value: Int

    def __add__(self, rhs: Int) -> Int:
        return self.value + rhs

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.value * 2, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostInt"


struct HostString(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = SIMD[.uint8, 4]
    var value: SIMD[.uint8, 4]
    # Deliberately absent from device_type so host and device layouts differ.
    var host_padding: SIMD[.uint8, 4]

    def __init__(out self, value: String) raises:
        if value.byte_length() < 4:
            raise Error("HostString requires at least four bytes")

        self.value = SIMD[.uint8, 4]()
        self.host_padding = SIMD[.uint8, 4]()
        comptime for i in range(4):
            self.value[i] = value.as_bytes()[i]

    def __add__(self, rhs: Self) -> Int:
        return Int(self.value[0]) + Int(rhs.value[0])

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        var device_value = self.value
        comptime for i in range(4):
            device_value[i] += 1
        encoder.encode(device_value, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostString"


def host_string_device_image(value: HostString) -> SIMD[.uint8, 4]:
    var device_value = value.value
    for i in range(4):
        device_value[i] += 1
    return device_value


def assert_simd_u8_4_equal(
    actual: SIMD[.uint8, 4], expected: SIMD[.uint8, 4]
) raises:
    for i in range(4):
        assert_equal(Int(actual[i]), Int(expected[i]))


def verify_foobar42_device_type_structure[DeviceClosureType: AnyType]() -> None:
    """Verify the compiler-synthesized `foobar42` `device_type` lit struct."""
    comptime r = reflect[DeviceClosureType]
    comptime assert r.is_struct()
    comptime assert r.field_count() == 2
    comptime assert r.field_index["capture2"]() == 0
    comptime assert r.field_index["anotherClosure"]() == 1
    comptime assert r.field["capture2"].T == Int

    comptime NestedDeviceType = r.field["anotherClosure"].T
    comptime nested = reflect[NestedDeviceType]
    comptime assert nested.is_struct()
    comptime assert nested.field_count() == 2
    comptime assert nested.field_index["capture1"]() == 0
    comptime assert nested.field_index["capture3"]() == 1
    comptime assert nested.field["capture1"].T == SIMD[.uint8, 4]
    comptime assert nested.field["capture3"].T == SIMD[.uint8, 4]


def verify_device_passable_interface[
    ClosureType: DevicePassable & def(Int) -> Int,
](
    closure: ClosureType,
    expected_capture2: Int,
    expected_nested_capture1: SIMD[.uint8, 4],
    expected_nested_capture3: SIMD[.uint8, 4],
) raises:
    comptime DeviceClosureType = ClosureType.device_type
    comptime assert DeviceClosureType != ClosureType
    comptime _verified_structure = (
        verify_foobar42_device_type_structure[DeviceClosureType]()
    )
    comptime assert ClosureType._is_convertible_to_device_type[
        DeviceClosureType
    ]()
    comptime assert not ClosureType._is_convertible_to_device_type[
        ClosureType
    ]()
    # The wrapper forwards this call to the closure storage's witness.
    assert_equal(ClosureType.get_type_name(), "foobar42")

    # Exercise host __call__ separately; device invocation is not supported yet.
    assert_equal(closure(4), 108)

    var encoded = alloc[DeviceClosureType]({length = 1}).unsafe_leak()
    var encoder = DefaultDeviceTypeEncoder()
    closure._to_device_type(encoder, encoded.unsafe_bitcast[NoneType]())

    comptime device_type = reflect[DeviceClosureType]
    ref capture2 = device_type.field_ref[0](encoded[])
    ref nested_value = device_type.field_ref[1](encoded[])
    comptime NestedDeviceType = device_type.field_at[1].T
    comptime nested_device_type = reflect[NestedDeviceType]

    assert_equal(rebind[Int](capture2), expected_capture2)
    assert_simd_u8_4_equal(
        rebind[SIMD[.uint8, 4]](nested_device_type.field_ref[0](nested_value)),
        expected_nested_capture1,
    )
    assert_simd_u8_4_equal(
        rebind[SIMD[.uint8, 4]](nested_device_type.field_ref[1](nested_value)),
        expected_nested_capture3,
    )
    encoded.unsafe_free()


def createDevicePassableClosures(
    capture1: HostString,
    capture2: HostInt,
    capture3: HostString,
) raises:
    def anotherClosure() {var} -> Int:
        return capture1 + capture3

    def foobar42(argument: Int) {var} -> Int:
        return capture2 + argument + anotherClosure()

    verify_device_passable_interface[type_of(foobar42)](
        foobar42,
        expected_capture2=capture2.value * 2,
        expected_nested_capture1=host_string_device_image(capture1),
        expected_nested_capture3=host_string_device_image(capture3),
    )


def main() raises:
    var capture1 = HostInt(value=atol(argv()[1]))
    var capture2 = HostString(String(argv()[2]))
    var capture3 = HostString(String(argv()[3]))
    createDevicePassableClosures(capture2, capture1, capture3)
