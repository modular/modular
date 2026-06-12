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
from std.gpu.host.device_context import DefaultDeviceTypeEncoder
from std.testing import assert_equal, TestSuite


# A DevicePassable type whose `_to_device_type` scales the encoded value, so a
# bit-copy is observably distinct from a proper dispatch to `_to_device_type`.
@fieldwise_init
struct ScaledInt(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = Int
    var raw: Int

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.raw * 2, target)

    @staticmethod
    def get_type_name() -> String:
        return "ScaledInt"


# A register-passable aggregate that is NOT itself `DevicePassable` but holds a
# `DevicePassable` member alongside a plain one.
@fieldwise_init
struct ScaledIntBox(ImplicitlyCopyable, TrivialRegisterPassable):
    var scaled: ScaledInt
    var tag: Int


# A second level of nesting: the `DevicePassable` member is reachable only
# transitively, through the `box` field.
@fieldwise_init
struct ScaledIntBoxBox(ImplicitlyCopyable, TrivialRegisterPassable):
    var box: ScaledIntBox
    var tag2: Int


# A register-passable aggregate with no `DevicePassable` member anywhere.
@fieldwise_init
struct PlainPair(ImplicitlyCopyable, TrivialRegisterPassable):
    var a: Int
    var b: Int


# `DefaultDeviceTypeEncoder.target()` is the current (host) target, so the
# device field layout matches the host layout and the encoded buffer can be
# read back through the host struct type.
def test_encode_fields_dispatches_device_passable_field() raises:
    var box = ScaledIntBox(scaled=ScaledInt(raw=7), tag=99)
    var buf = alloc[ScaledIntBox](1)
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_fields(box, buf.bitcast[NoneType]())
    # `scaled` is `DevicePassable`, so `ScaledInt._to_device_type` runs and
    # doubles `raw`; the plain `tag` field is bit-copied unchanged.
    assert_equal(buf[].scaled.raw, 14)
    assert_equal(buf[].tag, 99)
    buf.free()


def test_encode_fields_recurses_into_nested_composite() raises:
    var boxbox = ScaledIntBoxBox(
        box=ScaledIntBox(scaled=ScaledInt(raw=3), tag=10), tag2=200
    )
    var buf = alloc[ScaledIntBoxBox](1)
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_fields(boxbox, buf.bitcast[NoneType]())
    # `box` is not `DevicePassable` but transitively contains one, so the
    # recursion reaches `scaled` and doubles `raw`; the plain fields are
    # bit-copied unchanged.
    assert_equal(buf[].box.scaled.raw, 6)
    assert_equal(buf[].box.tag, 10)
    assert_equal(buf[].tag2, 200)
    buf.free()


def test_encode_fields_bit_copies_plain_fields() raises:
    var pair = PlainPair(a=11, b=22)
    var buf = alloc[PlainPair](1)
    var encoder = DefaultDeviceTypeEncoder()
    encoder.encode_fields(pair, buf.bitcast[NoneType]())
    # No field is `DevicePassable`, so every field is bit-copied unchanged.
    assert_equal(buf[].a, 11)
    assert_equal(buf[].b, 22)
    buf.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
