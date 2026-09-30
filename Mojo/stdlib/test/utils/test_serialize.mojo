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

from std.testing import TestSuite, assert_equal

from std.utils._serialize import _serialize, _serialize_elements


def _iota_ptr(
    data: List[Int32],
) -> OptionalPointer[Scalar[DType.int32], ImmutAnyOrigin]:
    return OptionalPointer[Scalar[DType.int32], ImmutAnyOrigin](
        data.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin]()
    )


def _iota(n: Int) -> List[Int32]:
    var data = List[Int32]()
    for i in range(n):
        data.append(Int32(i))
    return data^


def _write_tensor(
    ptr: OptionalPointer[Scalar[DType.int32], ImmutAnyOrigin],
    shape: List[Int],
    mut writer: Some[Writer],
):
    def serialize[T: Writable](val: T) {mut writer}:
        writer.write(val)

    _serialize[serialize_end_line=False](ptr, shape, serialize)


def test_serialize_elements() raises:
    var data = _iota(20)
    var out = String()

    def serialize[T: Writable](val: T) {mut out}:
        out.write(val)

    _serialize_elements(_iota_ptr(data), 5, serialize)
    assert_equal(out, "0, 1, 2, 3, 4")

    out = String()
    _serialize_elements[compact=True](_iota_ptr(data), 3, serialize)
    assert_equal(out, "[0, 1, 2]")

    out = String()
    _serialize_elements[compact=True](_iota_ptr(data), 20, serialize)
    assert_equal(out, "[0, 1, 2, ..., 17, 18, 19]")


def test_serialize_2d() raises:
    var data = _iota(6)
    var out = String()

    def serialize[T: Writable](val: T) {mut out}:
        out.write(val)

    _serialize(_iota_ptr(data), [2, 3], serialize)
    assert_equal(out, "[[0, 1, 2],\n[3, 4, 5]], dtype=int32, shape=2x3\n")


def test_serialize_3d_through_writer() raises:
    var data = _iota(4)
    var out = String()
    _write_tensor(_iota_ptr(data), [2, 1, 2], out)
    assert_equal(out, "[[[0, 1]],\n[[2, 3]]], dtype=int32, shape=2x1x2")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
