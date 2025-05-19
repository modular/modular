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
"""Provides layout and layout tensor types, which abstract memory layout for
multidimensional data.

- The [`Layout`](/mojo/stdlib/layout/layout/Layout) type represents a mapping
  between a set of logical coordinates and a linear index. It can be used, for
  example, to map logical tensor coordinates to a memory address, or to map GPU
  threads to tiles of data.

- The [`LayoutTensor`](/mojo/stdlib/layout/layout_tensor/LayoutTensor) type is a
  high-performance tensor with explicit memory layout via a `Layout`.
"""
from .int_tuple import UNKNOWN_VALUE, IntTuple
from .layout import Layout, LayoutList, composition, print_layout
from .layout_tensor import LayoutTensor, stack_allocation_like
from .runtime_layout import RuntimeLayout
from .runtime_tuple import RuntimeTuple
