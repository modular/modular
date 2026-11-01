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
bias; the weights still come from the unbiased scores. Selection, the
weight gather and the renorm run as main's ``moe_router_image_bias``
kernel (V2's router): ties at the top-6 cut resolve to the *lowest* expert
id, and a NaN score never wins a slot.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn.legacy.kernels import moe_router_image_bias
from max.nn.module import Module
from max.tensor import Tensor

from .mhc import claim_replicated

# The kernel keys its bias on a token id; the mask is mapped onto 0/1.
_IMAGE_ID = 1

_route_op = F.functional(moe_router_image_bias)


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


def route_logits(
    logits: Tensor,
    bias: Tensor,
    k: int,
    scaling: float,
    bias_vl: Tensor | None = None,
    image_token_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Scores float32 gate ``logits`` and routes each row to its top ``k``.

    The ``sqrtsoftplus`` scores, then main's router kernel for selection by
    ``scores + bias``, the unbiased weights' gather, their ``sum > 0``
    renorm and the ``scaling``. Image rows (``image_token_mask`` nonzero)
    select with ``bias_vl`` when one is given.

    Returns:
        ``(ids, weights)``: int32 and float32 ``[rows, k]``.
    """
    scores = _sqrt_softplus(logits)
    # NaN logits must stay NaN, as in the reference: the stable softplus
    # turns them into a finite score, which would let a nonzero bias steer
    # selection to arbitrary experts and hide the poisoned row behind
    # uniform weights.
    nan = F.constant(float("nan"), dtype=DType.float32, device=logits.mesh)
    scores = F.where(F.is_nan(logits), nan, scores)
    if image_token_mask is None or bias_vl is None:
        is_image = F.full(
            [logits.shape[0]], 0, dtype=DType.int32, device=scores.mapping
        )
        bias_vl = bias
    else:
        is_image = image_token_mask.cast(DType.bool).cast(DType.int32)
    ids, weights = _route_op(
        scores,
        bias,
        bias_vl,
        is_image,
        image_token_id=_IMAGE_ID,
        n_routed_experts=int(logits.shape[1]),
        n_experts_per_tok=k,
        routed_scaling_factor=scaling,
    )
    # The kernel op has no sharding rule: claim its rows' placement.
    return claim_replicated(ids, logits), claim_replicated(weights, logits)


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
        return route_logits(
            logits,
            self.bias,
            self.num_experts_per_tok,
            self.routed_scaling_factor,
            self.bias_vl,
            image_token_mask,
        )
