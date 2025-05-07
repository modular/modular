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

"""
Regex module for pattern matching.

This module provides functionality for searching and matching patterns in strings
using regular expressions. It implements a subset of regex features commonly found
in other languages like Python, JavaScript, and Perl.

The module supports:
- Pattern searching and matching with search(), match(), and fullmatch().
- String splitting with split().
- String substitution with sub().
- Case-sensitive and case-insensitive matching.
- Basic handling of escaped characters.

Example usage:

```mojo
from stdlib.regex import search, match, sub

# Search for a pattern
if result = search("world", "hello world"):
    print("Found at position:", result.value().start)

# Replace text
new_text = sub("hello", "hi", "hello world")  # "hi world"
```
"""

from .regex import compile, _match, search, fullmatch, split, sub
