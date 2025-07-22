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


struct MyList(Sized):
    var size: Int
    # ...

    fn __init__(out self):
        self.size = 0

    fn __len__(self) -> Int:
        return self.size


@fieldwise_init
struct IntLike(Intable):
    var i: Int

    fn __int__(self) -> Int:
        return self.i


@fieldwise_init
struct Dog(Copyable, Representable, Stringable, Writable):
    var name: String
    var age: Int

    # Allows the type to be written into any `Writer`
    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write("Dog(", self.name, ", ", self.age, ")")

    # Construct and return a `String` using the previous method
    fn __str__(self) -> String:
        return String.write(self)

    # Alternative full representation when calling `repr`
    fn __repr__(self) -> String:
        return String(
            "Dog(name=", repr(self.name), ", age=", repr(self.age), ")"
        )


def main():
    # Sized example
    print(len(MyList()))

    # Intable example
    value = IntLike(42)
    print(Int(value) == 42)

    # Stringable, Representable, Writable example
    dog = Dog("Rex", 5)
    print(repr(dog))
    print(dog)

    dog_info = String("String: {!s}\nRepresentation: {!r}").format(dog, dog)
    print(dog_info)
