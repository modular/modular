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

from std.testing import _assert_aborts, TestSuite


def test_resize_not_codepoint_boundary() raises:
    var s = String("😀longlonglonglong")

    def trigger() raises {s} -> None:
        s.resize(1)

    _assert_aborts(
        trigger,
        contains="does not lie on a codepoint boundary.",
    )


def test_resize_unsafe_not_codepoint_boundary() raises:
    var s = String("😀longlonglonglong")

    def trigger() raises {s} -> None:
        s.resize(unsafe_uninit_length=1)

    _assert_aborts(
        trigger,
        contains="does not lie on a codepoint boundary.",
    )


def test_resize_valid() raises:
    var s = String("😀😃")
    s.resize(4)
    s.resize(4)
    s.resize(5, 127)
    var s2 = String("😀😃")
    s2.resize(unsafe_uninit_length=4)
    s2.resize(unsafe_uninit_length=4)
    print("OK")


def test_resize_too_large() raises:
    var s = String("😀😃")

    def trigger() raises {s} -> None:
        s.resize(7)

    _assert_aborts(
        trigger,
        contains="does not lie on a codepoint boundary.",
    )


def test_resize_fill_byte_invalid() raises:
    var s = String()

    def trigger() raises {s} -> None:
        s.resize(10, 128)

    _assert_aborts(
        trigger,
        contains="Fill byte is the start of a multi-byte character.",
    )


def test_getitem_not_codepoint_boundary() raises:
    var s = String("😌😃")

    def trigger() raises {s} -> None:
        var _y = s[byte=1]

    _assert_aborts(
        trigger,
        contains="does not lie on a codepoint boundary.",
    )


def test_slice_not_codepoint_boundary() raises:
    var s = String("😌😃")

    def trigger() raises {s} -> None:
        var _y = s[byte=0:5]

    _assert_aborts(
        trigger,
        contains="is not a codepoint boundary.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
