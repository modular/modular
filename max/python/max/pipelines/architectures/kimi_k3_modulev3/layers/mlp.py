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
"""Kimi K3's SituAndMul activation and dense FFN (ModuleV3).

``tensor_parallel_dense_mlp`` shards an already-built layer in place.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh, NamedMapping
from max.experimental.tensor import Tensor

from .tensor_parallel import shard_fused_blocks


def kimi_situ_and_mul(
    gate_up: Tensor, beta: float, linear_beta: float | None, width: int
) -> Tensor:
    """Applies ``SituAndMul`` to a fused ``[gate ; up]`` tensor.

    Direct V3 port of ``kimi_k3/layers/mlp.py``'s ``kimi_situ_and_mul`` --
    same math (``beta * tanh(gate / beta) * sigmoid(gate)``, optionally
    ``linear_beta * tanh(up / linear_beta)`` on the up half), both halves
    upcast to fp32 for the activation and cast back at the end.

    Args:
        gate_up: Concatenation of the gate and up projections along the LAST
            axis, gate first, split at the midpoint.
        beta: ``config.activation_situ_beta``.
        linear_beta: ``config.activation_situ_linear_beta``; when ``None``
            the up half is passed through unchanged.
        width: The global FFN width, even under TP. ``split`` divides it
            across the shards; slicing would gather them.
    """
    out_dtype = gate_up.dtype
    gate_raw, up_raw = gate_up.split([width, width], axis=-1)
    gate = F.cast(gate_raw, DType.float32)
    up = F.cast(up_raw, DType.float32)
    situ = beta * F.tanh(gate / beta) * F.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * F.tanh(up / linear_beta)
    return F.cast(situ * up, out_dtype)


class KimiK3DenseMLP(Module[[Tensor], Tensor]):
    """The dense (non-MoE) FFN, ``KimiMLP`` with ``hidden_act="situ"``.

    Also used for K3's shared experts (``shared_intermediate_size =
    moe_intermediate_size * num_shared_experts``), which run on the original
    full-width identity, never in the MoE latent.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        beta: float,
        linear_beta: float | None,
    ) -> None:
        """Args:
        hidden_size: Model width.
        intermediate_size: FFN width.
        beta: ``config.activation_situ_beta``.
        linear_beta: ``config.activation_situ_linear_beta``.
        """
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta
        self.intermediate_size = intermediate_size
        # One fused projection rather than two: see `fuse_gate_up` in the V2
        # weight adapter, which concatenates the checkpoint's separate
        # gate_proj/up_proj at load time to match this layout.
        self.gate_up_proj = Linear(
            in_dim=hidden_size, out_dim=2 * intermediate_size, bias=False
        )
        self.down_proj = Linear(
            in_dim=intermediate_size, out_dim=hidden_size, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        activated = kimi_situ_and_mul(
            self.gate_up_proj(x),
            self.beta,
            self.linear_beta,
            self.intermediate_size,
        )
        return self.down_proj(activated)


def tensor_parallel_dense_mlp(
    layer: KimiK3DenseMLP, mesh: DeviceMesh
) -> KimiK3DenseMLP:
    """Shards a dense FFN along the ``tp`` mesh axis.

    ``down_proj`` is sharded on its input dim, so the caller must allreduce
    the output.
    """
    if mesh.axis_size(TP) == 1:
        return layer
    layer.gate_up_proj.weight = shard_fused_blocks(
        layer.gate_up_proj.weight, mesh, num_blocks=2, axis=0
    )
    layer.down_proj.weight = layer.down_proj.weight.to(
        NamedMapping(mesh, (None, TP))
    )
    return layer
