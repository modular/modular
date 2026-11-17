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
"""Deprecated alias of :mod:`max.nn.ep_config`."""

from max.nn.ep_config import (
    _EP_LOCAL_SYNC_RESERVED_WORDS,
    NUM_GROUPS,
    EPConfig,
    _ep_fused_moe_requested,
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)

__all__ = [
    "NUM_GROUPS",
    "_EP_LOCAL_SYNC_RESERVED_WORDS",
    "EPConfig",
    "_ep_fused_moe_requested",
    "calculate_ep_max_tokens_per_rank",
    "estimate_ep_memory_usage",
]
