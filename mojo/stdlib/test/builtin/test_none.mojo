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

from testing import assert_equal
from testing import TestSuite


def test_str():
    assert_equal(NoneType().__str__(), "None")


def test_repr():
    assert_equal(NoneType().__repr__(), "None")


def test_format_to():
    assert_equal(String.write(NoneType()), "None")


struct FromNone:
    var value: Int

    @implicit
    fn __init__(out self, none: NoneType):
        self.value = -1

    # FIXME: None literal should be of NoneType not !kgen.none.
    @always_inline
    @implicit
    fn __init__(out self, none: __mlir_type.`!kgen.none`):
        self = NoneType()

    @implicit
    fn __init__(out self, value: Int):
        self.value = value


def test_type_from_none():
    _ = FromNone(5)

    _ = FromNone(None)

    # -------------------------------------
    # Test implicit conversion from `None`
    # -------------------------------------

    fn foo(arg: FromNone):
        pass

    # FIXME:
    #   This currently fails, because it requires 2 "hops" of conversion:
    #       1. !kgen.none => NoneType
    #       2. NoneType => FromNone
    # foo(None)
    #
    #   But, interestingly, this does not fail?
    var _obj2: FromNone = None


def main():
    var suite = TestSuite()

    suite.test[test_str]()
    suite.test[test_repr]()
    suite.test[test_format_to]()
    suite.test[test_type_from_none]()

    suite^.run()
