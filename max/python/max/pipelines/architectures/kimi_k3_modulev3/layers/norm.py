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
"""Kimi K3's RMSNorm variants (ModuleV3): plain and gated.

K3's KDA layer needs ``o_norm``, fla's ``FusedRMSNormGated(activation="sigmoid")``
(``modeling_kimi_linear.py``'s import of ``fla.modules.FusedRMSNormGated``).
The real Triton kernel (``external-llm-servers/sglang/.../fla/fused_norm_gate.py``)
normalizes in fp32 (``b_x_hat = b_x * b_rstd``), multiplies by the weight
(``b_y = b_x_hat * b_w``), THEN multiplies by ``sigmoid(gate)``
(``b_y = b_y * tl.sigmoid(b_g)``), with a single cast to the output dtype at
the store. Weight-multiply strictly precedes the gate -- this is why K3's own
V2 code (``kimi_k3/layers/norm.py``) builds it from the general
``ops.rms_norm`` primitive rather than ``max.nn.legacy.state_space.gated_group_rmsnorm``,
whose kernel gates the INPUT (with ``silu``, before normalizing) rather than
the weighted output (with ``sigmoid``) -- a genuinely different operation, not
an interchangeable fused form of the same math.

This operates on whatever shape ``x``'s last axis is, matching
:class:`~max.nn.norm.RMSNorm`'s generic last-axis behavior: KDA's
``o_norm`` normalizes per-head over a ``[T, num_heads, head_dim]`` tensor with
a ``[head_dim]`` weight, broadcasting across ``num_heads`` for free, the same
way K3's V2 code does.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

# config.rms_norm_eps: input_layernorm, post_attention_layernorm,
# routed_expert_norm, o_norm, the residual norms and the final norm.
KIMI_RMS_NORM_EPS = 1e-5


def kimi_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Applies ``KimiRMSNorm``: cast-then-multiply, not Gemma-style.

    Direct V3 port of ``kimi_k3/layers/norm.py``'s ``kimi_rms_norm``: the
    same ``rms_norm`` primitive :func:`kimi_rms_norm_gated` builds on, with
    ``weight_offset=0.0`` (no Gemma-style ``+1`` gain) and
    ``multiply_before_cast=False`` (normalize, cast back to ``x``'s dtype,
    THEN multiply by weight -- not the reverse).
    """
    return rms_norm(
        x, weight, eps, weight_offset=0.0, multiply_before_cast=False
    )


def kimi_rms_norm_gated(
    x: Tensor, weight: Tensor, gate: Tensor, eps: float
) -> Tensor:
    """Applies fla's ``FusedRMSNormGated(activation="sigmoid")``.

    The whole expression runs in fp32 with one cast at the end, matching the
    Triton kernel: normalize, multiply by the weight, THEN multiply by
    ``sigmoid(gate)`` -- gating the WEIGHTED output, not the raw input.
    """
    out_dtype = x.dtype
    normed = rms_norm(
        F.cast(x, DType.float32),
        F.cast(weight, DType.float32),
        eps,
        weight_offset=0.0,
        multiply_before_cast=False,
    )
    gated = normed * F.sigmoid(F.cast(gate, DType.float32))
    return F.cast(gated, out_dtype)


class KimiRMSNormGated(Module[[Tensor, Tensor], Tensor]):
    """Owns ``o_norm``'s weight; see :func:`kimi_rms_norm_gated`."""

    weight: Tensor
    eps: float

    def __init__(self, dim: int, eps: float) -> None:
        self.weight = Tensor.ones([dim])
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        """Normalizes ``x`` and scales it by the gate."""
        return kimi_rms_norm_gated(x, self.weight, gate, self.eps)
