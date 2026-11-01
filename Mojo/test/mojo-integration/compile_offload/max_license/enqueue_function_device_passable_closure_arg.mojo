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

# RUN: %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys import align_of, size_of
from max.gpu.host import DeviceContext


# Host/device pair modeled on a DevicePassable buffer pair: a data pointer
# plus a runtime layout, with a distinct device_type that keeps the same
# canonical type.
@fieldwise_init
struct RuntimeLayout(ImplicitlyCopyable, TrivialRegisterPassable):
    var length: Int


@fieldwise_init
struct DeviceTensorSlice(
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    var ptr: MutPointer[Int32, MutUntrackedOrigin]
    var layout: RuntimeLayout

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DeviceTensorSlice"


@fieldwise_init
struct HostTensorSlice(
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    var ptr: MutPointer[Int32, MutUntrackedOrigin]
    var layout: RuntimeLayout

    comptime device_type: AnyType = DeviceTensorSlice

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(DeviceTensorSlice(self.ptr, self.layout), target)

    @staticmethod
    def get_type_name() -> String:
        return "HostTensorSlice"


# Nested-closure captures whose device_type is not Self. The kernel reads each
# capture as its host type, so host and device layouts must match.
@fieldwise_init
struct HostInt(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = Int
    var value: Int

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.value * 2, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostInt"


@fieldwise_init
struct HostString(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = SIMD[.int32, 4]
    var value: SIMD[.int32, 4]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.value + 1, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostString"


def apply_closure[
    closure_type: (def(Int32) -> None) & DevicePassable & ImplicitlyCopyable
](value: Int32, closure: closure_type):
    closure(value)


def apply_slice_closure[
    closure_type: (def() -> None) & DevicePassable & ImplicitlyCopyable
](closure: closure_type):
    closure()


def assert_closure_encodable[ClosureType: DevicePassable]():
    comptime assert ClosureType != ClosureType.device_type
    comptime assert not ClosureType._is_convertible_to_device_type[
        ClosureType
    ]()
    comptime assert ClosureType._is_implicitly_encodable_to[ClosureType]()


def assert_host_slice_distinct_and_encodable():
    comptime assert HostTensorSlice != HostTensorSlice.device_type
    comptime assert HostTensorSlice.device_type == DeviceTensorSlice
    comptime assert size_of[HostTensorSlice]() == size_of[DeviceTensorSlice]()
    comptime assert align_of[HostTensorSlice]() == align_of[DeviceTensorSlice]()
    comptime assert not HostTensorSlice._is_convertible_to_device_type[
        HostTensorSlice
    ]()
    comptime assert HostTensorSlice._is_convertible_to_device_type[
        DeviceTensorSlice
    ]()
    comptime assert HostTensorSlice._is_implicitly_encodable_to[
        DeviceTensorSlice
    ]()


def main() raises:
    var ctx = DeviceContext()
    var captured = Int32(42)

    def identity_capture(value: Int32) {var captured}:
        _ = captured + value

    assert_closure_encodable[type_of(identity_capture)]()

    comptime identity_kernel = apply_closure[type_of(identity_capture)]
    var compiled_identity = ctx.compile_function[identity_kernel]()
    ctx.enqueue_function(
        compiled_identity,
        Int32(1),
        identity_capture,
        grid_dim=1,
        block_dim=1,
    )

    assert_host_slice_distinct_and_encodable()
    var slice = HostTensorSlice(
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=1),
        RuntimeLayout(0),
    )

    def slice_capture() {var slice}:
        _ = slice.layout.length

    assert_closure_encodable[type_of(slice_capture)]()

    comptime slice_kernel = apply_slice_closure[type_of(slice_capture)]
    var compiled_slice = ctx.compile_function[slice_kernel]()
    ctx.enqueue_function(
        compiled_slice,
        slice_capture,
        grid_dim=1,
        block_dim=1,
    )

    var capture1 = HostString(SIMD[.int32, 4](1))
    var capture2 = HostInt(21)
    var capture3 = HostString(SIMD[.int32, 4](3))

    def another_closure(argument: Int32) {var}:
        _ = capture1
        _ = capture3
        _ = argument

    def captures_closure(argument: Int32) {var}:
        _ = capture2
        another_closure(argument)

    assert_closure_encodable[type_of(another_closure)]()
    assert_closure_encodable[type_of(captures_closure)]()

    comptime nested_kernel = apply_closure[type_of(captures_closure)]
    var compiled_nested = ctx.compile_function[nested_kernel]()
    ctx.enqueue_function(
        compiled_nested,
        Int32(1),
        captures_closure,
        grid_dim=1,
        block_dim=1,
    )
