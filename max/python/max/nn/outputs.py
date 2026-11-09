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

"""Options that select which outputs a transformer model returns."""

from enum import Enum


class ReturnLogits(str, Enum):
    """Which logits a transformer model returns per sequence.

    ``LAST_TOKEN`` returns only the final position's logits, ``VARIABLE``
    returns ``return_n_logits`` trailing positions, and ``ALL`` returns
    every position.
    """

    LAST_TOKEN = "last_token"
    """Return only the final token's logits."""

    VARIABLE = "variable"
    """Return ``return_n_logits`` trailing positions' logits."""

    ALL = "all"
    """Return every token's logits."""


class ReturnHiddenStates(str, Enum):
    """Which hidden states a transformer model returns per sequence.

    ``NONE`` returns none, ``LAST`` returns the final token's hidden
    states, and ``ALL`` returns every token's. See each member's
    description for the other modes.
    """

    NONE = "none"
    """Return no hidden states."""

    LAST = "last"
    """Return the final token's hidden states."""

    # Like LAST but returns the post-allgather per-device tensors rather
    # than collapsing to a single copy. Lets callers consume the data that
    # was already placed on every device by the allgather, avoiding a
    # redundant broadcast.
    LAST_PER_DEVICE = "last_per_device"
    """Return the final token's hidden states as per-device tensors,
    one per device, without collapsing identical replicas."""

    ALL_NORMALIZED = "all_normalized"
    """Return every token's hidden states after the model's final norm."""

    ALL = "all"
    """Return every token's hidden states."""

    SELECTED_LAYERS = "selected_layers"
    """Return hidden states captured at specific layer indices."""
