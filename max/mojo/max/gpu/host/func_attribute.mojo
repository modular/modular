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
"""
GPU Kernel Function Attributes Module.

This module provides structures for defining and managing GPU kernel function attributes.
It implements functionality similar to CUDA's CUfunction_attribute enum, allowing
for querying and setting various attributes that control kernel execution behavior
and resource allocation.

The module includes:
- `Attribute`: A value type representing different GPU kernel function attribute types
- `FuncAttribute`: A structure that pairs an attribute type with its value

These structures enable fine-grained control over GPU kernel execution parameters
such as shared memory allocation, cache behavior, and cluster configuration.
"""


@__doc_inline
from ._legacy.func_attribute import (
    Attribute,
    FuncAttribute,
)
