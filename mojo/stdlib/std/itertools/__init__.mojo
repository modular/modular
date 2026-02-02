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
"""Iterator tools: count, cycle, product, repeat for lazy sequence generation.

The `itertools` package provides utilities for creating and composing iterators
for efficient lazy evaluation. It offers building blocks for generating infinite
sequences, computing Cartesian products, cycling through elements, and repeating
elements without materializing entire collections in memory. These tools enable
functional programming patterns and memory-efficient iteration over large or
infinite sequences.

Use this package for generating sequences without explicit loops, creating
combinations of elements from multiple collections, or implementing functional
iteration patterns. These tools are particularly useful for nested loops,
grid-based computations, or any scenario requiring efficient lazy evaluation.
"""

from .itertools import count, cycle, product, repeat
