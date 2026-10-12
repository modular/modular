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
"""Mask configuration for attention."""

from __future__ import annotations

from enum import Enum


class MHAMaskVariant(str, Enum):
    """Defines the integer mask variant codes used by multihead attention kernels."""

    CAUSAL_MASK = 0
    """Causal masking; each position attends only to itself and earlier positions."""
    NULL_MASK = 2
    """No masking; every position attends to every position."""
    CHUNKED_CAUSAL_MASK = 3
    """Causal masking applied over fixed-size chunks of the sequence."""
    SLIDING_WINDOW_CAUSAL_MASK = 4
    """Each position attends to itself and the previous ``window_size - 1``
    positions."""
    SLIDING_WINDOW_NONCAUSAL_MASK = 5
    """Each position attends to itself, the previous ``window_size - 1``
    positions, and every later position."""


class AttentionMaskVariant(str, Enum):
    """Defines the string mask variant identifiers used in attention configuration."""

    NULL_MASK = "null"
    """No masking; every position attends to every position."""
    CAUSAL_MASK = "causal"
    """Causal masking; each position attends only to itself and earlier positions."""
    TENSOR_MASK = "tensor_mask"
    """An explicit mask tensor supplied by the caller."""
    CHUNKED_CAUSAL_MASK = "chunked_causal"
    """Causal masking applied over fixed-size chunks of the sequence."""
    SLIDING_WINDOW_CAUSAL_MASK = "sliding_window_causal"
    """Each position attends to itself and the previous ``window_size - 1``
    positions."""
    SLIDING_WINDOW_NONCAUSAL_MASK = "sliding_window_noncausal"
    """Each position attends to itself, the previous ``window_size - 1``
    positions, and every later position."""
