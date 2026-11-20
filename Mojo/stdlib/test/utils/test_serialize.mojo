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


def _iota(n: Int) -> List[Int32]:
    var data = List[Int32]()
    for i in range(n):
        data.append(Int32(i))
    return data^


def _write_tensor(
    ptr: Pointer[Scalar[DType.int32], _],
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

    _serialize_elements(data.unsafe_ptr(), 5, serialize)
    assert_equal(out, "0, 1, 2, 3, 4")

    out = String()
    _serialize_elements[compact=True](data.unsafe_ptr(), 3, serialize)
    assert_equal(out, "[0, 1, 2]")

    out = String()
    _serialize_elements[compact=True](data.unsafe_ptr(), 20, serialize)
    assert_equal(out, "[0, 1, 2, ..., 17, 18, 19]")


def test_serialize_2d() raises:
    var data = _iota(6)
    var out = String()

    def serialize[T: Writable](val: T) {mut out}:
        out.write(val)

    _serialize(data.unsafe_ptr(), [2, 3], serialize)
    assert_equal(out, "[[0, 1, 2],\n[3, 4, 5]], dtype=int32, shape=2x3\n")


def test_serialize_3d_through_writer() raises:
    var data = _iota(4)
    var out = String()
    _write_tensor(data.unsafe_ptr(), [2, 1, 2], out)
    assert_equal(out, "[[[0, 1]],\n[[2, 3]]], dtype=int32, shape=2x1x2")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
