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
"""The DeepSeek-V4.1 ``sqrtsoftplus`` ``noaux_tc`` MoE router (ModuleV3).

384 routed experts, top-6, fp32 scores. The correction bias steers expert
*selection* only -- the routing weights are gathered from the unbiased
scores ("noaux_tc"). Image-span rows swap ``bias_vl`` in for the selection
bias; the weights still come from the unbiased scores. Selection ties at
the top-6 cut resolve to the *lowest* expert id (``ops.top_k``/``argsort``
tie order is unspecified, so the selection loop is written out explicitly).
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module import Module

# The graph-level tie-break loop needs python-level k unrolling, so the
# supported top-k is fixed at V4.1's 6 (config `num_experts_per_tok`).
_SUPPORTED_TOPK = 6


def _stable_top_k(scores: Tensor, k: int, num_experts: int) -> Tensor:
    """Selects the ``k`` largest scores per row, ties to the lowest expert id.

    The reference's per-row selection loop: each slot takes the row max,
    and among equal maxima the smallest expert id wins. The picked expert
    is masked out of range after each pick, so slot order is descending
    score. ``ops.top_k`` cannot be used directly because its tie order is
    unspecified, and the contract is lowest-id.

    Args:
        scores: fp32 selection scores, ``[rows, num_experts]``. NaNs are
            clamped to ``-1e30`` before selection: an unclamped NaN would
            make the row max NaN, leave ``is_max`` all-false, and push the
            id out of range (NaNs must never win a slot).
        k: Experts to pick per row.
        num_experts: Size of the expert axis.

    Returns:
        The selected expert ids, int32 ``[rows, k]``, in pick order
        (highest score first; among tied maxima, lowest id first). The
        caller gathers the routing weights from the unbiased scores.
    """
    # Every constant and range takes ``scores``' mesh: a ``.device`` read
    # raises once ``scores`` spans devices under TP, and a bare device
    # would strand the constant off the mesh (the window math's idiom).
    mesh = scores.mesh
    nan_clamp = F.constant(-1e30, dtype=DType.float32, device=mesh)
    # The picked-expert mask must sit strictly below the NaN clamp: with
    # a shared floor, an all-NaN row (every score clamped to -1e30)
    # would re-pick the id tied at the clamp for all six slots; with
    # -inf it yields six DISTINCT ascending ids, matching the -inf mask
    # for already-picked experts.
    picked_mask = F.constant(float("-inf"), dtype=DType.float32, device=mesh)
    current = F.where(F.is_nan(scores), nan_clamp, scores)
    expert_axis = F.arange(0, num_experts, dtype=DType.int32, device=mesh)
    ids_parts: list[Tensor] = []
    for _ in range(k):
        # F reductions keep the reduced axis at size 1: max/min below
        # return [rows, 1], so the ids are squeezed back to [rows].
        max_value = F.max(current, axis=-1)
        # Tied maxima: the smallest expert id wins.
        is_max = F.equal(current, max_value)
        candidate = F.where(
            is_max,
            expert_axis.cast(DType.float32),
            F.full(
                [num_experts],
                float(num_experts),
                dtype=DType.float32,
                device=mesh,
            ),
        )
        expert_id = F.min(candidate, axis=-1)
        ids_i = expert_id.cast(DType.int32).squeeze(-1)
        ids_parts.append(ids_i)
        # Mask the picked expert out for the remaining slots.
        picked = F.equal(expert_axis, ids_i.unsqueeze(-1))
        current = F.where(picked, picked_mask, current)
    return F.stack(ids_parts, axis=1)


def _sqrt_softplus(logits: Tensor) -> Tensor:
    """``sqrt(softplus(logits))`` in fp32, stable for very negative logits.

    ``log(1+exp(x))`` rounds to 0 in fp32 for ``x <= -16.64`` and ``exp``
    overflows above ~20; ``max(x,0)+log1p(exp(-|x|))`` is exact everywhere
    and matches the reference's linear branch above 20. A NaN logit does
    not survive ``max``/``abs`` here, so the caller restores it.
    """
    zero = F.constant(0.0, dtype=DType.float32, device=logits.mesh)
    softplus = F.elementwise_max(logits, zero) + F.log1p(
        F.exp(F.abs(logits) * -1.0)
    )
    return F.sqrt(softplus)


class Router(Module[[Tensor, Tensor | None], tuple[Tensor, Tensor]]):
    """The V4.1 MoE gate, mounted at ``ffn.gate``.

    Parameters are the checkpoint's ``ffn.gate.{weight, bias}`` (plus
    ``bias_vl`` when the model has vision):
    ``weight`` bf16 ``[n_routed_experts, hidden]``, ``bias``/``bias_vl`` f32
    ``[n_routed_experts]``. ``bias_vl`` is *optional* -- text-only serving
    skips it (a checkpoint without ``vision_config`` never creates it), and
    in that case ``image_token_mask`` is ignored.

    The shared expert is always applied by the caller; this module only
    routes the experts.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 384,
        num_experts_per_tok: int = 6,
        routed_scaling_factor: float = 1.5,
        with_bias_vl: bool = False,
    ) -> None:
        """Initializes the router's parameters.

        Args:
            hidden_dim: The hidden state width (5120).
            num_experts: The number of routed experts (384).
            num_experts_per_tok: Experts selected per token (6).
            routed_scaling_factor: The post-renorm weight scale (1.5).
            with_bias_vl: Whether to hold the vision bias. On for all 40
                trunk layers of the vision checkpoint; off for text-only.
        """
        super().__init__()
        if num_experts_per_tok != _SUPPORTED_TOPK:
            raise ValueError(
                f"top-k {num_experts_per_tok} unsupported: the stable "
                f"selection loop is fixed at k={_SUPPORTED_TOPK}"
            )
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor

        # ``self.device`` defaults to CPU (Module.device), so a fresh
        # module never mounts the accelerator: a param on the default
        # device drags the whole graph there on hosts without a GPU
        # toolchain (the pipeline reassigns these under ``to``/state
        # load, matching every other ModuleV3 layer).
        device = self.device
        self.weight = Tensor.zeros(
            [num_experts, hidden_dim],
            dtype=DType.bfloat16,
            device=device,
        )
        self.bias = Tensor.zeros(
            [num_experts], dtype=DType.float32, device=device
        )
        self.bias_vl = (
            Tensor.zeros([num_experts], dtype=DType.float32, device=device)
            if with_bias_vl
            else None
        )

    def forward(
        self, x: Tensor, image_token_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Routes ``x`` to its top-6 experts.

        Args:
            x: The collapsed sublayer input, ``[rows, hidden_dim]`` (the
                caller has already applied its RMSNorm).
            image_token_mask: Optional uint8 ``[rows]`` row mask, nonzero on
                image-span tokens. When this router holds a
                ``bias_vl``, those rows select with ``bias_vl`` instead of
                ``bias`` -- selection only, never the weights.

        Returns:
            ``(topk_ids, topk_weights)``: global expert ids int32
            ``[rows, 6]`` and routing weights fp32 ``[rows, 6]``, already
            normalized (renorm with the ``sum>0`` floor) and scaled by
            ``routed_scaling_factor``. NaN scores never win a slot, so ids
            are always in ``[0, num_experts)``.
        """
        logits = x.cast(DType.float32) @ self.weight.cast(DType.float32).T
        scores = _sqrt_softplus(logits)
        # NaN logits must stay NaN, as in the reference: the stable
        # softplus above turns them into a finite score, which would let
        # a nonzero bias steer selection to arbitrary experts and hide
        # the poisoned row behind uniform weights.
        nan = F.constant(float("nan"), dtype=DType.float32, device=logits.mesh)
        scores = F.where(F.is_nan(logits), nan, scores)

        ids = _stable_top_k(
            scores + self._selection_bias(image_token_mask),
            self.num_experts_per_tok,
            self.num_experts,
        )
        # The routing weights come from the UNBIASED scores (noaux_tc).
        weights = self._gather_weights(scores, ids)

        # Renorm with the sum>0 floor: an additive 1e-20 instead would
        # differ only when every weight underflows, yielding ~0 rather
        # than exactly 0.
        weight_sum = F.sum(weights, axis=-1)
        one = F.constant(1.0, dtype=DType.float32, device=weights.mesh)
        denom = F.where(weight_sum > 0.0, weight_sum, one)
        weights = weights * (self.routed_scaling_factor / denom)
        return ids, weights

    @staticmethod
    def _gather_weights(scores: Tensor, ids: Tensor) -> Tensor:
        """Gathers each row's ``k`` picked scores, ``[rows, 6]``.

        ``F.gather`` splices the indices' shape into the output
        (:func:`~max.graph.ops.gather`'s shape rule), so a ``[rows, 6]``
        id tensor against ``[rows, num_experts]`` scores would yield
        ``[rows, rows, 6]``. ``gather_nd`` with ``batch_dims=1`` indexes
        per row instead: indices ``[rows, 6, 1]`` against ``[rows,
        num_experts]`` gathers exactly ``[rows, 6]``.
        """
        return F.gather_nd(scores, ids.unsqueeze(-1), batch_dims=1)

    def _selection_bias(self, image_token_mask: Tensor | None) -> Tensor:
        """The per-row selection bias, swapping ``bias_vl`` in on image rows."""
        if image_token_mask is None or self.bias_vl is None:
            return self.bias
        is_image = image_token_mask.cast(DType.bool)
        return F.where(is_image.unsqueeze(-1), self.bias_vl, self.bias)
