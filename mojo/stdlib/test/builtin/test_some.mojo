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


@fieldwise_init
struct Foo[z: Int]:
    pass


@fieldwise_init
struct Bar[x: Int, //, y: Int, *, foo: Foo[x], bar: Foo[y] = Foo[y]()](
    Intable, ImplicitlyCopyable
):
    fn __int__(self) -> Int:
        return self.x + self.y + self.foo.z + self.bar.z


fn takes_some_arg(x: Some[Intable]) -> Int:
    return x.__int__()


def test_some_arg():
    assert_equal(takes_some_arg(Bar[2, foo = Foo[4]()]()), 12)
    assert_equal(takes_some_arg(Bar[foo = Foo[5](), y=6]()), 22)
    assert_equal(takes_some_arg(Bar[foo = Foo[5](), bar = Foo[7]()]()), 24)


fn takes_some_param[x: Some[Intable]]() -> Int:
    return materialize[x]().__int__()


def test_some_param():
    assert_equal(takes_some_param[Bar[2, foo = Foo[4]()]()](), 12)
    assert_equal(takes_some_param[Bar[foo = Foo[5](), y=6]()](), 22)
    assert_equal(takes_some_param[Bar[foo = Foo[5](), bar = Foo[7]()]()](), 24)


fn takes_multiple_traits(x: Some[Intable & Copyable]) -> __type_of(x):
    return x.copy()


def test_some_return():
    assert_equal(takes_multiple_traits(Bar[2, foo = Foo[4]()]()).__int__(), 12)


def test_closure():
    fn some_closure(x: Some[Intable]) -> Int:
        return x.__int__() * 2

    fn takes_some_closure[func: fn (Some[Intable]) -> Int]() raises:
        assert_equal(func(Int(4)), 8)

    takes_some_closure[some_closure]()


def main():
    var suite = TestSuite()

    suite.test[test_some_arg]()
    suite.test[test_some_param]()
    suite.test[test_some_return]()
    suite.test[test_closure]()

    suite^.run()
