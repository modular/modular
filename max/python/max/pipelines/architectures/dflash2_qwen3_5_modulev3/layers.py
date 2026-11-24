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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/dflash2_qwen3_5/layers.py
# at main 4df8111afba.
"""The norm DFlash2 trained with, and the two modules it adds to DFlash.

The convolution and the selector are transcribed from the authors' own vLLM
fork (``vllm/model_executor/models/qwen3_dflash2.py``): ``_grouped_conv`` /
``DFlashGroupedConv`` and ``_score_edges`` / ``CandidateSelector``.
"""

from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.tensor import Tensor

__all__ = ["DFlash2CandidateSelector", "DFlash2GroupedConv", "DFlash2RMSNorm"]


class DFlash2RMSNorm(Module[[Tensor], Tensor]):
    """RMS normalization over the last axis, scaled before the cast back.

    Scales by ``weight``, in the input's dtype.
    """

    def __init__(self, dim: int, *, eps: float) -> None:
        self.weight = Tensor.ones([dim])
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x`` normalized and scaled, in ``x``'s dtype."""
        return F.rms_norm(
            x,
            self.weight.cast(x.dtype),
            self.eps,
            weight_offset=0.0,
            multiply_before_cast=True,
        )


class DFlash2GroupedConv(Module[..., Tensor]):
    """One sublayer's pair of dynamic grouped depthwise convolutions.

    Applied along the *block* axis, once before the sublayer and once after
    it::

        out[t] = sum_tap (base_kernel[side, tap] + delta[t, tap, group(c)])
                 * x[t - tap]

    ``base_kernel`` is per channel; ``delta`` is per group of ``group_size``
    channels. ``t`` is block-local, so a tap reading before the block's first
    row reads zero and the convolution never crosses a block boundary.

    ``kernel_projection`` runs once, on the sublayer's input, and emits both
    sides' deltas: :meth:`prepare` applies side 0 and returns side 1's for
    :meth:`finish`.

    No ``forward``: a layer calls :meth:`prepare` and :meth:`finish`.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        taps: int,
        group_size: int,
        block_size: int,
    ) -> None:
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide"
                f" hidden_size={hidden_size}."
            )
        if not 1 <= taps <= block_size:
            raise ValueError(
                f"conv_kernel_size={taps} must be in [1, {block_size}], the"
                " block it convolves within."
            )
        self.hidden_size = hidden_size
        self.taps = taps
        self.group_size = group_size
        self.block_size = block_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = Tensor.zeros([2, taps, hidden_size])
        self.kernel_projection = Linear(
            hidden_size, 2 * taps * self.num_groups, bias=False
        )

    def prepare(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Applies the pre-sublayer convolution.

        Args:
            x: ``[rows, hidden]``, one dense ``block_size``-row block per
                sequence.

        Returns:
            The convolved rows, and the ``[rows, taps, groups]`` deltas
            :meth:`finish` applies.
        """
        coefficients = self.kernel_projection(x).reshape(
            [-1, 2, self.taps, self.num_groups]
        )
        return (
            self._convolve(x, coefficients[:, 0], side=0),
            coefficients[:, 1],
        )

    def finish(self, x: Tensor, coefficients: Tensor) -> Tensor:
        """Applies the post-sublayer convolution with :meth:`prepare`'s deltas."""
        return self._convolve(x, coefficients, side=1)

    def _convolve(self, x: Tensor, delta: Tensor, *, side: int) -> Tensor:
        block, taps = self.block_size, self.taps
        groups, group_size = self.num_groups, self.group_size
        # The block axis is a reshape of the row axis, so the zero taps at a
        # block's first rows are structural.
        blocks = x.reshape([-1, block, groups, group_size])
        base = (
            self.base_kernel[side]
            .cast(blocks.dtype)
            .reshape([1, 1, taps, groups, group_size])
        )
        coefficients = base + delta.reshape(
            [-1, block, taps, groups]
        ).unsqueeze(-1)
        out = coefficients[:, :, 0] * blocks
        for tap in range(1, taps):
            zeros = F.broadcast_to(
                F.constant(0, blocks.dtype, device=blocks.device),
                [blocks.shape[0], tap, groups, group_size],
            )
            shifted = F.concat([zeros, blocks[:, :-tap]], axis=1)
            out = out + coefficients[:, :, tap] * shifted
        return out.reshape([-1, self.hidden_size])


class DFlash2CandidateSelector(Module[..., Tensor]):
    """Low-rank bilinear scorer over adjacent candidate tokens, and its walk.

    ``S[l, p, c] = unary[l, c] + <pred_book[pred_id[l, p]] * proj(h_l),
    succ_book[cand_id[l, c]]>``, where ``pred_id`` at slot ``l`` is slot
    ``l - 1``'s candidates and, at ``l == 0``, the anchor (the last verified
    token) broadcast over the predecessor axis.

    No ``forward``: the proposer calls :meth:`score_edges` and
    :meth:`select_path`.
    """

    def __init__(
        self, hidden_size: int, *, vocab_size: int, rank: int, top_k: int
    ) -> None:
        self.top_k = top_k
        self.predecessor_codebook = Tensor.zeros([vocab_size, rank])
        self.successor_codebook = Tensor.zeros([vocab_size, rank])
        self.hidden_projection = Linear(hidden_size, rank, bias=False)

    def score_edges(
        self,
        candidate_ids: Tensor,
        unary_logits: Tensor,
        hidden_states: Tensor,
        anchor_token_ids: Tensor,
    ) -> Tensor:
        """Scores every adjacent candidate pair in the block.

        Args:
            candidate_ids: ``[batch, steps, top_k]`` top-k ids per mask slot.
            unary_logits: ``[batch, steps, top_k]`` their head logits.
            hidden_states: ``[batch, steps, hidden]`` drafter output per slot.
            anchor_token_ids: ``[batch]`` id of the last verified token.

        Returns:
            ``[batch, steps, top_k, top_k]`` indexed ``[b, l, p, c]``.
        """
        top_k = self.top_k
        hidden = self.hidden_projection(hidden_states)
        successors = F.gather(self.successor_codebook, candidate_ids, axis=0)
        anchor_row = F.broadcast_to(
            anchor_token_ids.reshape([-1, 1, 1]),
            [candidate_ids.shape[0], 1, top_k],
        )
        predecessor_ids = F.concat([anchor_row, candidate_ids[:, :-1]], axis=1)
        predecessors = F.gather(
            self.predecessor_codebook, predecessor_ids, axis=0
        )
        edges = F.matmul(
            predecessors * hidden.unsqueeze(2), successors.transpose(-1, -2)
        )
        return unary_logits.unsqueeze(2) + edges

    @staticmethod
    def select_path(
        scores: Tensor,
        candidate_ids: Tensor,
        sample: Callable[[Tensor, int, Tensor], Tensor],
    ) -> Tensor:
        """Walks the scores left to right, following the chosen successor.

        A single pass, unrolled over the static slot axis. Slot 0's
        predecessor row is the anchor broadcast over ``top_k``, so seeding
        the walk at index 0 reads the same row as any other index.

        Args:
            scores: ``[batch, steps, top_k, top_k]`` from :meth:`score_edges`.
            candidate_ids: ``[batch, steps, top_k]``.
            sample: Draws step ``l``'s ``[batch]`` candidate index from its
                ``[batch, top_k]`` successor scores and the candidates' ids.

        Returns:
            ``[batch, steps]`` chosen token ids.
        """
        steps = int(scores.shape[1])
        previous = F.broadcast_to(
            F.constant(0, DType.int64, device=scores.device),
            [scores.shape[0], 1],
        )
        tokens: list[Tensor] = []
        for step in range(steps):
            row = F.gather_nd(scores[:, step], previous, batch_dims=1)
            previous = (
                sample(row, step, candidate_ids[:, step])
                .cast(DType.int64)
                .unsqueeze(-1)
            )
            tokens.append(
                F.gather_nd(candidate_ids[:, step], previous, batch_dims=1)
            )
        return F.stack(tokens, axis=1)
