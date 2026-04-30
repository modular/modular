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

- Unified type and struct reflection via `reflect[T]()` returning a
  `Reflected[T]` handle (use `r.name()` and `r.base_name()` for type names)
- Function name and linkage introspection (`get_function_name`, `get_linkage_name`)
- Source location introspection (`source_location`, `call_location`)
- Deprecated free functions: `get_type_name` (use `reflect[T]().name()`) and
  `get_base_type_name` (use `reflect[T]().base_name()`)

`reflect` is auto-imported via the prelude. The other names listed above
must be imported explicitly from `std.reflection`.

Example:
```mojo
struct Point:
    var x: Int
    var y: Float64

def print_fields[T: AnyType]():
    comptime r = reflect[T]()
    comptime names = r.field_names()
    comptime for i in range(r.field_count()):
        print(names[i])

def main():
    print_fields[Point]()  # Prints: x, y
```
"""

from .location import SourceLocation, source_location, call_location
from .reflect import Reflected, reflect
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
    struct_field_ref,
    struct_field_types,
    struct_field_index_by_name,
    struct_field_type_by_name,
    offset_of,
    ReflectedType,
)
