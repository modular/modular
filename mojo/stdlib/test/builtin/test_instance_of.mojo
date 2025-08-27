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


@fieldwise_init
struct Foo[z: Int]:
    pass


@fieldwise_init
struct Bar[x: Int, //, y: Int, *, foo: Foo[x], bar: Foo[y] = Foo[y]()](
    Intable, Copyable
):
    fn __int__(self) -> Int:
        return self.x + self.y + self.foo.z + self.bar.z


fn takes_instance_of_arg(x: Some[Intable]) -> Int:
    return x.__int__()


def test_instance_of_arg():
    assert_equal(takes_instance_of_arg(Bar[2, foo = Foo[4]()]()), 12)
    assert_equal(takes_instance_of_arg(Bar[foo = Foo[5](), y=6]()), 22)
    assert_equal(
        takes_instance_of_arg(Bar[foo = Foo[5](), bar = Foo[7]()]()), 24
    )


fn takes_instance_of_param[x: Some[Intable]]() -> Int:
    return x.__int__()


def test_instance_of_param():
    assert_equal(takes_instance_of_param[Bar[2, foo = Foo[4]()]()](), 12)
    assert_equal(takes_instance_of_param[Bar[foo = Foo[5](), y=6]()](), 22)
    assert_equal(
        takes_instance_of_param[Bar[foo = Foo[5](), bar = Foo[7]()]()](), 24
    )


fn takes_multiple_traits(x: Some[Intable & Copyable]) -> __type_of(x):
    return x


def test_instance_of_return():
    assert_equal(takes_multiple_traits(Bar[2, foo = Foo[4]()]()).__int__(), 12)


def main():
    test_instance_of_arg()
    test_instance_of_param()
    test_instance_of_return()
