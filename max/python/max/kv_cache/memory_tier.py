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

"""Memory tier enum for the paged KV cache."""

from __future__ import annotations

from enum import IntEnum


class MemoryTier(IntEnum):
    """Memory tier enum for the paged KV cache."""

    MEMORY_TIER_UNSPECIFIED = 0
    MEMORY_TIER_GPU = 1
    MEMORY_TIER_CPU = 2
    MEMORY_TIER_DISK = 3
    MEMORY_TIER_OBJECT_STORE = 4
