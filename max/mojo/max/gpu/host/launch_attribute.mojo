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
GPU Launch Attributes for Kernel Execution Control.

This module provides structures for configuring GPU kernel execution through launch attributes.
It implements a Mojo interface to CUDA's launch attribute system, allowing fine-grained control
over kernel execution characteristics such as memory access policies, synchronization behavior,
cluster dimensions, and resource allocation.

The main components include:
- `LaunchAttributeID`: Identifies different types of launch attributes
- `LaunchAttributeValue`: Stores the value for a specific attribute type
- `LaunchAttribute`: Combines an ID and value to form a complete attribute
- `AccessPolicyWindow`: Configures memory access patterns and caching behavior
- `AccessProperty`: Defines cache persistence properties for memory access

These structures enable optimizing GPU kernel performance by controlling execution parameters
at a granular level, similar to CUDA's native launch attribute system.
"""


@__doc_inline
from ._legacy.launch_attribute import (
    AccessPolicyWindow,
    AccessProperty,
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)
