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

"""Muse Spark residual gate for the ModuleV3 API."""

from __future__ import annotations

from dataclasses import dataclass, replace

from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module
from max.tensor import Tensor

from .moe import MuseSparkMoEBranch, decode_cond, shard_types
from .norms import CenteredRMSNorm, InputScaledRMSNorm, WeightlessRMSNorm

# The only activations `gated_residual_rms_norm` takes: Muse Spark's.
_FUSED_DTYPE, _FUSED_WIDTH = DType.bfloat16, 8192


class MuseSparkResidualGate(Module[[Tensor, Tensor], Tensor]):
    """Mixes the residual and the branch output with a learned per-channel gate.

    ``out = alpha * residual + beta * h`` with ``beta = sigmoid(gate / T)``
    and ``alpha = sqrt(max(sigmoid(-gate / T) * (1 + beta), eps))``.
    """

    def __init__(
        self, hidden_size: int, temperature: float, eps: float = 1e-3
    ) -> None:
        super().__init__()
        self.gate = Tensor.zeros([hidden_size])
        self.temperature = temperature
        self.eps = eps

    def forward(self, residual: Tensor, h: Tensor) -> Tensor:
        """Returns the residual scaled by the learned gate."""
        # The reference keeps the gate and the mix in fp32.
        w = self.gate.cast(DType.float32) / self.temperature
        beta = F.sigmoid(w)
        alpha = F.sqrt(F.max(F.sigmoid(-w) * (1.0 + beta), self.eps))
        out = alpha * residual.cast(DType.float32) + beta * h.cast(
            DType.float32
        )
        return out.cast(residual.dtype)


def _gated_residual_rms_norm(
    branch: TensorValue,
    residual: TensorValue,
    post_weight: TensorValue,
    gate: TensorValue,
    norm_weight: TensorValue,
    scale_branch: bool,
    post_eps: float,
    norm_eps: float,
    temperature: float,
    gate_eps: float,
) -> list[TensorValue]:
    return [
        out.tensor
        for out in ops.custom(
            "gated_residual_rms_norm",
            device=branch.device,
            values=[
                branch,
                residual,
                post_weight,
                gate,
                norm_weight,
                *(
                    ops.constant(v, DType.float32, device=DeviceRef.CPU())
                    for v in (post_eps, norm_eps, temperature, gate_eps)
                ),
            ],
            out_types=[branch.type, branch.type],
            parameters={"scale_branch": scale_branch},
        )
    ]


gated_residual_rms_norm = F.functional(_gated_residual_rms_norm)


@dataclass
class MuseSparkResidualBoundary:
    """A branch not yet joined to the residual stream.

    The next pre-norm joins it: one launch does the branch's post-norm,
    the gate's mix and the pre-norm, as vLLM's ``dual_norm`` does.
    """

    residual: Tensor
    branch: Tensor | MuseSparkMoEBranch
    post_norm: WeightlessRMSNorm | InputScaledRMSNorm
    gate: MuseSparkResidualGate

    def join(self) -> Tensor:
        """Returns the new residual stream."""
        return self.gate(self.residual, self.post_norm(self._whole_branch()))

    def join_and_norm(self, norm: CenteredRMSNorm) -> tuple[Tensor, Tensor]:
        """Returns the new residual stream and ``norm`` of it."""
        if not all(
            isinstance(d, Accelerator) for d in self.residual.mesh.devices
        ):
            h = self.join()
            return h, norm(h)
        if isinstance(self.branch, MuseSparkMoEBranch):
            return self._join_expert_rows(norm)
        if (
            self.branch.dtype != _FUSED_DTYPE
            or norm.weight.dtype != _FUSED_DTYPE
            or self.gate.gate.dtype != _FUSED_DTYPE
            or self.branch.shape[-1] != _FUSED_WIDTH
        ):
            h = self.join()
            return h, norm(h)
        post_norm = self.post_norm
        scaled = isinstance(post_norm, InputScaledRMSNorm)
        h, normed = gated_residual_rms_norm(
            self.branch,
            self.residual,
            post_norm.weight
            if isinstance(post_norm, InputScaledRMSNorm)
            else norm.weight,
            self.gate.gate,
            norm.weight,
            scaled,
            self.post_norm.eps,
            norm.eps,
            self.gate.temperature,
            self.gate.eps,
        )
        mapping = self.residual.mapping
        return h.rebind_mapping(mapping), normed.rebind_mapping(mapping)

    def _whole_branch(self) -> Tensor:
        if isinstance(self.branch, MuseSparkMoEBranch):
            return self.branch.whole()
        return self.branch

    def _join_expert_rows(self, norm: CenteredRMSNorm) -> tuple[Tensor, Tensor]:
        """Joins the MoE branch.

        A decode step gathers the projection's output columns inside the
        fused join.
        """
        moe = self.branch
        assert isinstance(moe, MuseSparkMoEBranch)
        assert isinstance(self.post_norm, InputScaledRMSNorm)
        post_norm, gate = self.post_norm, self.gate

        def sharded() -> tuple[Tensor, Tensor]:
            return F.allgather_gated_residual_rms_norm(
                moe.columns(),
                self.residual,
                post_norm.weight,
                gate.gate,
                norm.weight,
                post_norm.eps,
                norm.eps,
                gate.temperature,
                gate.eps,
                scale_branch=True,
            )

        spec = (self.residual.mapping, shard_types(self.residual))
        h, normed = decode_cond(
            self.residual.shape[0],
            [spec, spec],
            sharded,
            lambda: replace(self, branch=moe.exchange()).join_and_norm(norm),
            [*moe.weights, post_norm.weight, gate.gate, norm.weight],
        )
        return h, normed
