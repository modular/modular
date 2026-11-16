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
"""What a KDA layer's verify keeps for the rollback to replay its conv."""

from __future__ import annotations

from typing import NamedTuple

from max.tensor import Tensor

__all__ = ["KdaReplayInputs"]


class KdaReplayInputs(NamedTuple):
    """One KDA layer's per-token inputs to the two state kernels.

    Speculative decoding runs the verify pass over K+1 positions and then has
    to land the pools on the accepted prefix instead. Every op feeding these
    tensors is causal or pointwise, so re-running the kernels over the accepted
    rows from the pre-verify state reproduces the state the verify pass held at
    that length -- the argument Qwen3.5's rollback in
    ``speculative/gated_delta_rollback.py`` rests on.

    What does not carry over from Qwen3.5 is the shape of ``raw_gate``: KDA's
    forget gate is one value per *channel* per head, not one scalar per head,
    so a rollback that replays a ``[total_tokens, num_heads]`` decay is
    replaying the wrong tensor.
    """

    qkv: Tensor
    """``[total_tokens, conv_dim]`` conv input, pre-convolution."""

    conv_weight: Tensor
    """``[conv_dim, conv_kernel_size]`` depthwise weights."""

    raw_gate: Tensor
    """``[total_tokens, num_heads, head_dim]`` forget-gate pre-activation.

    Per channel. Pre-``dt_bias`` and pre-activation, matching what the
    recurrence op consumes.
    """

    beta_logits: Tensor
    """``[total_tokens, num_heads]`` input-gate logits, pre-sigmoid."""

    qkv_window: Tensor | None = None
    """``[batch_size * (K + 1), conv_dim]`` copy of each row the verify
    window holds, one slot per row, that a verify's conv kernel writes when
    there are drafts, or ``None`` outside a verify. The rollback's fold places
    each conv window from it."""
