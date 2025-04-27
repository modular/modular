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
"""The string package provides comprehensive Unicode string handling functionality for Mojo.

This package implements Unicode-aware string types and operations, with UTF-8 support.
It includes efficient implementations for string manipulation, formatting, and Unicode
operations while maintaining memory safety and performance.

Key Components:
- `String`: The main string type supporting UTF-8 encoded text
- `StringSlice`: Memory-efficient string view type for zero-copy operations
- `InlineString`: Small string optimization for short strings
- `Codepoint`: Unicode code point handling and operations
- Format: String formatting and interpolation utilities

Core Features:
- Unicode support with UTF-8 encoding
- Efficient string slicing and views
- String formatting and interpolation
- Memory-safe string operations
- Unicode case conversion
- Unicode property lookups and validation

Example:
```mojo
    # Basic string creation and manipulation
    var s = String("Hello, 世界")
    var slice = s[0:5] # "Hello"

    # Unicode-aware operations
    for c in s.codepoints():
        print(c.to_uppercase())

    # String formatting
    var name = "Mojo"
    var formatted = String("Hello, {name}!")
```

Note:

String stores data using UTF-8, and all operations (unless clearly noted) are intended to
be fully Unicode compliant and maintain correct UTF-8 encoded data.
A handful of operations are known to not be Unicode / UTF-8 compliant yet, but will be
fixed as time permits.
"""

from .codepoint import Codepoint
from .string import String, ascii, atof, atol, chr, ord
from .string_slice import CodepointsIter, StaticString, StringSlice
