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

# Metal-only test: when a unified closure captures `DevicePointer`s (the
# register-passable, `DevicePassable` borrow of a `DeviceBuffer`), encoding the
# closure with a `MetalDeviceTypeEncoder` must register every referenced
# buffer's handle in `MetalDeviceTypeEncoder._buffers`. The Metal launch path
# relies on that list to bind the buffers a kernel touches.
#
# A closure cannot capture a `DeviceBuffer` directly: `DeviceBuffer` is
# memory-only, so the closure would not be `RegisterPassableTrivial` and would
# never receive the synthesized `DevicePassable` conformance. `DevicePointer`
# is `TrivialRegisterPassable`, and its `_to_device_type` dispatches to
# `encode_device_ptr`, which appends the owning buffer's handle.

from std.gpu.host.device_context import DeviceContext, DevicePointer
from std.gpu.host._device_context_metal import MetalDeviceTypeEncoder
from std.testing import assert_equal, assert_true, TestSuite


# A register-passable aggregate that is NOT itself `DevicePassable` but holds
# `DevicePassable` `DevicePointer` members. Encoding a closure that captures it
# must recurse through the struct into each pointer (the behavior under test).
@fieldwise_init
struct PtrPair[
    first_mut: Bool,
    second_mut: Bool,
    //,
    first_origin: Origin[mut=first_mut],
    second_origin: Origin[mut=second_mut],
](ImplicitlyCopyable, TrivialRegisterPassable):
    var first: DevicePointer[DType.float32, Self.first_origin]
    var second: DevicePointer[DType.float32, Self.second_origin]


def test_closure_registers_captured_buffers() raises:
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](16)
    var b = ctx.enqueue_create_buffer[DType.float32](32)
    var pa = a.device_ptr()
    var pb = b.device_ptr()

    # `pa`/`pb` are captured by value; the body uses them so they are really
    # captured. The closure is encoded, never executed.
    def k(z: Int) {var pa, var pb} -> Int:
        return pa.offset() + pb.offset()

    var storage = alloc[type_of(k)](1)
    var encoder = MetalDeviceTypeEncoder()
    k._to_device_type(
        encoder, storage.bitcast[NoneType]().as_unsafe_any_origin()
    )

    # Both captured pointers route through `encode_device_ptr`, registering
    # their owning buffers' handles — and nothing else.
    assert_equal(len(encoder._buffers), 2)

    var handle_a = a._handle.value()
    var handle_b = b._handle.value()
    var found_a = False
    var found_b = False
    for ref handle in encoder._buffers:
        if handle == handle_a:
            found_a = True
        if handle == handle_b:
            found_b = True
    assert_true(found_a, "buffer a was not registered in _buffers")
    assert_true(found_b, "buffer b was not registered in _buffers")

    storage.free()


def test_closure_registers_buffers_via_nested_struct() raises:
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](16)
    var b = ctx.enqueue_create_buffer[DType.float32](32)
    var pair = PtrPair(a.device_ptr(), b.device_ptr())

    # The closure captures a register-passable struct that only transitively
    # contains `DevicePassable` members; `encode_fields` must recurse into it.
    def k(z: Int) {var pair} -> Int:
        return pair.first.offset() + pair.second.offset()

    var storage = alloc[type_of(k)](1)
    var encoder = MetalDeviceTypeEncoder()
    k._to_device_type(
        encoder, storage.bitcast[NoneType]().as_unsafe_any_origin()
    )

    assert_equal(len(encoder._buffers), 2)

    var handle_a = a._handle.value()
    var handle_b = b._handle.value()
    var found_a = False
    var found_b = False
    for ref handle in encoder._buffers:
        if handle == handle_a:
            found_a = True
        if handle == handle_b:
            found_b = True
    assert_true(found_a, "nested buffer a was not registered in _buffers")
    assert_true(found_b, "nested buffer b was not registered in _buffers")

    storage.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
