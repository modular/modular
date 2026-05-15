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
from std.testing import assert_true


def my_generic_fn1[T: AnyType](value: T):
    pass


def my_generic_fn2(value: Some[AnyType]):
    pass


def foo1[T: Intable, //](x: T) -> Int:
    return x.__int__()


def foo2(x: Some[Intable]) -> Int:
    return x.__int__()


def sync_parallelize1[
    FuncType: def(Int) -> None,
](func: FuncType):
    pass


def sync_parallelize2(func: Some[def(Int) -> None]):
    pass


def int_to_none(x: Int) -> None:
    pass


def show1[*Ts: Writable](*pack: *Ts):
    pass


def show2(*pack: *SomeTypeList[Writable]):
    pass


@fieldwise_init
struct Struct1[T: Copyable & ImplicitlyDestructible]:
    var x: Self.T

    def __getitem__[I: Indexer, //](self, idx: I) -> ref[self.x] Self.T:
        return self.x


@fieldwise_init
struct Struct2[T: Copyable & ImplicitlyDestructible]:
    var x: Self.T

    def __getitem__(self, idx: Some[Indexer]) -> ref[self.x] Self.T:
        return self.x


def main() raises:
    my_generic_fn1(42)
    my_generic_fn2(42)

    assert_true(foo1(42.1) == 42)
    assert_true(foo2(42.1) == 42)

    sync_parallelize1(int_to_none)
    sync_parallelize2(int_to_none)

    show1(1, 2, 3)
    show2(1, 2, 3)

    var s1 = Struct1(42)
    var s2 = Struct2(42)
    assert_true(s1[0] == 42)
    assert_true(s2[0] == 42)
