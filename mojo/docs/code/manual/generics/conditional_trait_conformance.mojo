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

comptime BaseTraits = Copyable & ImplicitlyDestructible


@fieldwise_init
struct Wrapper[T: BaseTraits](
    Boolable where conforms_to(T, Boolable),
    Writable where conforms_to(T, Writable),
):
    var value: Self.T

    fn __bool__(self) -> Bool where conforms_to(Self.T, Boolable):
        return trait_downcast[Boolable](self.value).__bool__()


@fieldwise_init
struct NotWritable(BaseTraits):
    var data: Int


@fieldwise_init
struct Pair[L: BaseTraits, R: BaseTraits](
    Hashable where conforms_to(L, Hashable) and conforms_to(R, Hashable)
):
    var left: Self.L
    var right: Self.R


@fieldwise_init
struct NotHashable(BaseTraits):
    var data: Int


@fieldwise_init
struct Foo[T: AnyType](Copyable, Writable where conforms_to(T, Writable)):
    pass


def main() raises:
    var w_int = Wrapper[Int](42)
    print(w_int)  # Wrapper[Int](value=42)

    var w_str = Wrapper[String]("Hello")
    print(w_str)  # Wrapper[String](value=Hello)

    var w_empty_str = Wrapper[String]("")
    if w_str:
        print(t"Non-empty string \"{w_str.value}\" is truthy")
    else:
        print(t"Empty string \"{w_str.value}\" is falsy")
    if w_empty_str:
        print(t"Non-empty string \"{w_empty_str.value}\" is truthy")
    else:
        print(t"Empty string \"{w_empty_str.value}\" is falsy")

    var w_not_writable = Wrapper[NotWritable](
        NotWritable(10)
    )  # This is constructable
    # Does not compile
    # Error: constraint declared (for `__bool__()` function) evaluated to
    # False, expected 'Bool(conforms_to(T, std::builtin::anytype::AnyType &
    # std::builtin::anytype::ImplicitlyDestructible &
    # std::builtin::bool::Boolable))'
    # if w_not_writable:
    #     print(t"NotWritable with data {w_not_writable.value.data} is truthy")
    # else:
    #     print(t"NotWritable with data {w_not_writable.value.data} is falsy")

    _ = w_not_writable  # Avoid unused variable warning
    # print(w_not_writable) #  Compile time error
    # invalid call to 'print': could not convert element of 'values' with
    # type 'Wrapper[NotWritable]' to expected type 'Writable'

    var pair = Pair[Int, String](left=1, right="one")
    var hash = hash(pair)
    print(hash)  # Prints the hash of the pair

    var pair2 = Pair[Int, NotHashable](
        left=1, right=NotHashable(10)
    )  # Constructable
    _ = pair2  # Avoid unused variable warning
    # var hash2 = hash(pair2) # Compile time error
