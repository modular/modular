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
"""Compatibility wrapper for gpu.block module.

This module has been moved to gpu.primitives.block. This file provides
backward compatibility for existing code that imports from gpu.block.

DEPRECATED: Import from gpu.primitives.block instead.
"""

# Re-export all functions from the new location
from .primitives.block import (
    broadcast,
    max,
    min,
    prefix_sum,
    sum,
)
