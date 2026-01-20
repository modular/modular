# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from testing import assert_equal, TestSuite


def raise_an_error():
    raise Error("MojoError: This is an error!")


def test_error_raising():
    try:
        raise_an_error()
    except e:
        assert_equal(String(e), "MojoError: This is an error!")


def test_from_and_to_string():
    var my_string: String = "FOO"
    var error = Error(my_string)
    assert_equal(String(error), "FOO")

    assert_equal(String(Error("bad")), "bad")
    assert_equal(repr(Error("err")), "Error('err')")


struct TestError[origin: MutOrigin](Copyable, Writable):
    var record: Pointer[List[String], Self.origin]

    fn __init__(out self, ref [Self.origin]record: List[String]):
        self.record = Pointer(to=record)

    fn write_to(self, mut writer: Some[Writer]):
        self.record[].append("write_to")
        writer.write("TestError")

    fn __del__(deinit self):
        self.record[].append("__del__")

    fn __copyinit__(out self, other: Self):
        self.record = other.record
        self.record[].append("__copyinit__")

    fn __moveinit__(out self, deinit other: Self):
        self.record = other.record
        self.record[].append("__moveinit__")


def test_destroys_erased_error():
    var record = List[String]()
    var error = Error(TestError(record))
    assert_equal(record, ["__moveinit__"])
    _ = error^
    assert_equal(record, ["__moveinit__", "__del__"])


def test_copy_and_move_erased_error():
    var record = List[String]()
    var error = Error(TestError(record))
    assert_equal(record, ["__moveinit__"])
    var error2 = error.copy()
    # No `__copyinit__` because the error is behind an `ArcPointer`.
    assert_equal(record, ["__moveinit__"])
    _ = error2^
    assert_equal(record, ["__moveinit__"])
    _ = error^
    assert_equal(record, ["__moveinit__", "__del__"])


def test_write_to_erased_error():
    var record = List[String]()
    var error = Error(TestError(record))
    assert_equal(record, ["__moveinit__"])

    var writer = String()
    error.write_to(writer)
    assert_equal(writer, "TestError")
    assert_equal(record, ["__moveinit__", "write_to", "__del__"])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
