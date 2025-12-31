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
"""Compile-time reflection utilities for introspecting Mojo types and functions.

This module provides compile-time reflection capabilities including:

- Type name introspection (`get_type_name`)
- Function name and linkage introspection (`get_function_name`, `get_linkage_name`)
- Struct field reflection (`struct_field_count`, `struct_field_names`, `struct_field_types`)
- Field lookup by name (`struct_field_index_by_name`, `struct_field_type_by_name`)

Example:
```mojo
from reflection import struct_field_count, struct_field_names

struct Point:
    var x: Int
    var y: Float64

fn print_fields[T: AnyType]():
    comptime names = struct_field_names[T]()
    @parameter
    for i in range(struct_field_count[T]()):
        print(names[i])

fn main():
    print_fields[Point]()  # Prints: x, y
```
"""

from .reflection import (
    # Type and function name introspection
    get_linkage_name,
    get_function_name,
    get_type_name,
    # Struct field reflection (works with generics)
    struct_field_count,
    struct_field_names,
    struct_field_types,
    # Field lookup by name
    struct_field_index_by_name,
    struct_field_type_by_name,
    # Wrapper types for reflection results
    ReflectedType,
)
