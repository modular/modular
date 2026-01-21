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

from reflection import (
    struct_field_count,
    struct_field_names,
    get_type_name,
    struct_field_types,
)


fn show_type[T: AnyType]():
    """Displays a compile-time representation of a struct's fields and types.

    Prints the struct name followed by a tree view of its fields with their
    types. Uses box-drawing characters for visual structure. Does not recurse
    into nested struct types or handle indentation for nested structures.

    Parameters:
        T: The struct type to inspect.

    Constraints:
        T must be a struct type with reflectable fields.

    Examples:
    ```mojo
            struct Point:
                var x: Int
                var y: Float64

            show_type[Point]()

            # Output:
            # struct Point
            # ├── var x: Int
            # └── var y: Float64
    ```
    """
    comptime type_name = get_type_name[T]()
    comptime field_count = struct_field_count[T]()
    comptime field_names = struct_field_names[T]()
    comptime field_types = struct_field_types[T]()
    print("struct", type_name)

    @parameter
    for idx in range(field_count):
        comptime field_name = field_names[idx]
        comptime field_type = get_type_name[field_types[idx]]()
        var intro = "├──" if idx < (field_count - 1) else "└──"
        print(intro, " var ", field_name, ": ", field_type, sep="")

    print()


@fieldwise_init
struct MyStruct:
    """A simple example struct demonstrating reflection capabilities.

    Contains a String field and an Optional Int field for testing
    struct field introspection and type reflection.
    """

    var x: String
    var y: Optional[Int]


comptime DefaultItemCount = 10


struct ParameterizedStruct[T: Copyable, item_count: Int = DefaultItemCount](
    Copyable
):
    var list: List[Self.T]

    fn __init__(out self):
        self.list = List[Self.T](capacity=Self.item_count)


fn main():
    show_type[MyStruct]()
    show_type[Optional[Float64]]()
    show_type[Dict[Int, String]]()
    show_type[ParameterizedStruct[String, item_count=5]]()
