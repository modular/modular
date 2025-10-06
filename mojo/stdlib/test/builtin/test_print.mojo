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

from tempfile import NamedTemporaryFile

from builtin._location import __call_location, _SourceLocation
from testing import TestSuite

from utils import IndexList


@always_inline
fn _assert_error[T: Writable](msg: T, loc: _SourceLocation) -> String:
    return loc.prefix(String("AssertionError: ", msg))


fn _assert_equal_error(
    lhs: String, rhs: String, msg: String, loc: _SourceLocation
) -> String:
    var err = (
        "`left == right` comparison failed:\n   left: "
        + lhs
        + "\n  right: "
        + rhs
    )
    if msg:
        err += "\n  reason: " + msg
    return _assert_error(err, loc)


struct PrintChecker(Movable):
    var tmp: NamedTemporaryFile
    var cursor: UInt64
    var call_location: _SourceLocation

    @always_inline
    fn __init__(out self) raises:
        self.tmp = NamedTemporaryFile("rw")
        self.call_location = __call_location()
        self.cursor = 0

    fn __enter__(var self) -> Self:
        return self^

    fn stream(self) -> FileDescriptor:
        return FileDescriptor(self.tmp._file_handle._get_raw_fd())

    fn check_line(mut self, expected: String, msg: String = "") raises:
        print(end="", file=self.stream(), flush=True)
        _ = self.tmp.seek(self.cursor)
        var result = self.tmp.read()[:-1]
        if result != expected:
            raise _assert_equal_error(result, expected, msg, self.call_location)
        self.cursor += len(result) + 1

    fn check_line_starts_with(
        mut self, prefix: String, msg: String = ""
    ) raises:
        print(end="", file=self.stream(), flush=True)
        _ = self.tmp.seek(self.cursor)
        var result = self.tmp.read()[:-1]
        var prefix_len = len(prefix)
        if len(result) < prefix_len:
            raise _assert_error(msg, self.call_location)
        if result[:prefix_len] != prefix:
            raise _assert_equal_error(
                result[:prefix_len], prefix, msg, self.call_location
            )
        self.cursor += len(result) + 1


def test_print():
    with PrintChecker() as checker:
        print("Hello", file=checker.stream())
        checker.check_line("Hello")

        print("World", flush=True, file=checker.stream())
        checker.check_line("World")

        var hello: StaticString = "Hello,"
        var world: String = "world!"
        var f: Bool = False
        print(">", hello, world, 42, True, f, file=checker.stream())
        checker.check_line("> Hello, world! 42 True False")

        var float32: Float32 = 99.9
        var float64: Float64 = -129.2901823
        print(">", 3.14, file=checker.stream())
        checker.check_line("> 3.14")
        print(">", float32, file=checker.stream())
        checker.check_line("> 99.9")
        print(">", float64, file=checker.stream())
        checker.check_line("> -129.2901823")
        print(">", IndexList[3](1, 2, 3), file=checker.stream())
        checker.check_line_starts_with("> (1, 2, 3)")

        print(">", 9223372036854775806, file=checker.stream())
        checker.check_line("> 9223372036854775806")

        var pi = 3.1415916535897743
        print(">", pi, file=checker.stream())
        checker.check_line("> 3.1415916535897743")
        var x = (pi - 3.141591) * 1e6
        print(">", x, file=checker.stream())
        checker.check_line_starts_with("> 0.6535")

        print("Hello world", file=checker.stream())
        checker.check_line("Hello world")


def test_print_end():
    with PrintChecker() as checker:
        print("Hello", end=" World\n", file=checker.stream())
        checker.check_line("Hello World")


def test_print_sep():
    with PrintChecker() as checker:
        print("a", "b", "c", sep="/", file=checker.stream())
        checker.check_line("a/b/c")

        print("a", 1, 2, sep="/", end="xx\n", file=checker.stream())
        checker.check_line("a/1/2xx")


def main():
    var suite = TestSuite()

    suite.test[test_print]()
    suite.test[test_print_end]()
    suite.test[test_print_sep]()

    suite^.run()
