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


fn takes_intable(x: InstanceOf[Intable]) -> Int:
    return x.__int__()


@fieldwise_init
struct Foo[z: Int]:
    pass


@fieldwise_init
struct Bar[x: Int, //, y: Int, *, foo: Foo[x], bar: Foo[y] = Foo[y]()](Intable):
    fn __int__(self) -> Int:
        return self.x + self.y + self.foo.z + self.bar.z


def test_instance_of():
    assert_equal(takes_intable(Bar[2, foo = Foo[4]()]()), 12)
    assert_equal(takes_intable(Bar[foo = Foo[5](), y=6]()), 22)
    assert_equal(takes_intable(Bar[foo = Foo[5](), bar = Foo[7]()]()), 24)


def main():
    test_instance_of()
