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
"""Compile-time reflection utilities for introspecting Mojo types and functions.

This module provides compile-time reflection capabilities including:

- Type name introspection (`get_type_name`)
- Function name and linkage introspection (`get_function_name`, `get_linkage_name`)
- Type checking (`is_struct_type`)
- Struct field reflection (`struct_field_count`, `struct_field_names`, `struct_field_types`)
- Field lookup by name (`struct_field_index_by_name`, `struct_field_type_by_name`)
- Field offset calculation (`offset_of`)
- Source location introspection (`source_location`, `call_location`)
- Base type reflection (`get_base_type_name`)

Example:
```mojo
from reflection import struct_field_count, struct_field_names

struct Point:
    var x: Int
    var y: Float64

fn print_fields[T: AnyType]():
    comptime names = struct_field_names[T]()
    comptime for i in range(struct_field_count[T]()):
        print(names[i])

fn main():
    print_fields[Point]()  # Prints: x, y
```
"""

from .location import SourceLocation, source_location, call_location
from .type_info import (
    get_linkage_name,
    get_function_name,
    get_type_name,
    # Base type reflection (for parameterized types)
    get_base_type_name,
)
from .struct_fields import (
    is_struct_type,
    struct_field_count,
    struct_field_names,
    struct_field_types,
    struct_field_index_by_name,
    struct_field_type_by_name,
    offset_of,
    ReflectedType,
)
