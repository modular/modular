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
"""The two modules DFlash2 adds to the DFlash block drafter, on ModuleV3.

The math is the graph-API Qwen3.5 drafter's
(:mod:`~max.pipelines.architectures.dflash2_qwen3_5.layers`), which
transcribes the authors' vLLM fork. The convolution accumulates in float32
and rounds once, as vLLM's Beam kernels do.
"""

from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn import Linear, Module
from max.tensor import Tensor

__all__ = ["DFlash2BeamCandidateSelector", "DFlash2BeamConv"]


class DFlash2BeamConv(Module[..., Tensor]):
    """One sublayer's pair of dynamic grouped depthwise convolutions.

    Applied along the block axis, before the sublayer and after it::

        out[t] = sum_tap (base_kernel[side, tap] + delta[t, tap, group(c)])
                 * x[t - tap]

    ``t`` is block-local, so a tap reading before the block's first row reads
    zero. ``kernel_projection`` runs once, on the sublayer's input, and emits
    both sides' deltas: :meth:`prepare` applies side 0 and returns side 1's
    for :meth:`finish`.

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
        # block's first rows are structural: the shifted term below starts at
        # row `tap` of every block.
        blocks = x.cast(DType.float32).reshape([-1, block, groups, group_size])
        base = (
            self.base_kernel[side]
            .cast(DType.float32)
            .reshape([1, 1, taps, groups, group_size])
        )
        coefficients = base + delta.cast(DType.float32).reshape(
            [-1, block, taps, groups, 1]
        )
        out = coefficients[:, :, 0] * blocks
        for tap in range(1, taps):
            shifted = (
                out[:, tap:] + coefficients[:, tap:, tap] * blocks[:, :-tap]
            )
            out = F.concat([out[:, :tap], shifted], axis=1)
        return out.reshape([-1, self.hidden_size]).cast(x.dtype)


class DFlash2BeamCandidateSelector(Module[..., Tensor]):
    """Low-rank bilinear scorer over adjacent candidates, and its walk.

    ``S[l, p, c] = unary[l, c] + <pred_book[pred_id[l, p]] * proj(h_l),
    succ_book[cand_id[l, c]]>``, where ``pred_id`` at slot ``l`` is slot
    ``l - 1``'s candidates and, at ``l == 0``, the anchor broadcast over the
    predecessor axis.

    The codebooks and the projection are replicated; the scoring runs on one
    device's copy, where the drafts are sampled.
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
        candidate_ids: TensorValue,
        unary_logits: TensorValue,
        projected: TensorValue,
        anchor_token_ids: TensorValue,
        device: int = 0,
    ) -> TensorValue:
        """Scores every adjacent candidate pair in the block.

        Args:
            candidate_ids: ``[batch, steps, top_k]`` candidate ids per slot.
            unary_logits: ``[batch, steps, top_k]`` float32 head logits.
            projected: ``[batch, steps, rank]``, the drafter's output at
                each slot through :attr:`hidden_projection`.
            anchor_token_ids: ``[batch]`` id of each block's anchor.
            device: Which device's codebooks to read.

        Returns:
            ``[batch, steps, top_k, top_k]`` float32, indexed ``[b, l, p, c]``.
        """
        top_k = self.top_k
        successor_book = TensorValue(
            self.successor_codebook.local_shards[device]
        )
        predecessor_book = TensorValue(
            self.predecessor_codebook.local_shards[device]
        )
        successors = ops.gather(successor_book, candidate_ids, axis=0)
        anchor_row = ops.broadcast_to(
            anchor_token_ids.reshape((-1, 1, 1)),
            [candidate_ids.shape[0], 1, top_k],
        )
        predecessor_ids = ops.concat(
            [anchor_row, candidate_ids[:, :-1]], axis=1
        )
        predecessors = ops.gather(predecessor_book, predecessor_ids, axis=0)
        # The product rounds to the codebooks' dtype before the dot, which
        # accumulates in float32 and rounds once, as vLLM's kernel does.
        edges = ops.matmul(
            predecessors * ops.unsqueeze(projected, 2),
            ops.transpose(successors, -1, -2),
        )
        return ops.unsqueeze(unary_logits, 2) + ops.cast(edges, DType.float32)

    @staticmethod
    def select_path(
        scores: TensorValue,
        candidate_ids: TensorValue,
        sampler: Callable[[TensorValue, int], TensorValue],
    ) -> TensorValue:
        """Walks the scores left to right, following the chosen successor.

        A single pass, unrolled over the static slot axis. Slot 0's
        predecessor row is the anchor broadcast over ``top_k``, so seeding
        the walk at index 0 reads the same row as any other index.

        Args:
            scores: ``[batch, steps, top_k, top_k]`` from :meth:`score_edges`.
            candidate_ids: ``[batch, steps, top_k]``.
            sampler: Draws step ``l``'s ``[batch]`` candidate index from its
                ``[batch, top_k]`` successor scores.

        Returns:
            ``[batch, steps]`` chosen token ids.
        """
        steps = int(scores.shape[1])
        previous = ops.broadcast_to(
            ops.constant(0, DType.int64, device=scores.device),
            [scores.shape[0], 1],
        )
        tokens: list[TensorValue] = []
        for step in range(steps):
            row = ops.gather_nd(scores[:, step], previous, batch_dims=1)
            previous = ops.unsqueeze(
                ops.cast(sampler(row, step), DType.int64), axis=-1
            )
            tokens.append(
                ops.gather_nd(candidate_ids[:, step], previous, batch_dims=1)
            )
        return ops.stack(tokens, axis=1)
