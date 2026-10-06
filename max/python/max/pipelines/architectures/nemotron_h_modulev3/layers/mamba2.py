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
"""The Nemotron-H Mamba-2 mixer."""

from __future__ import annotations

from max import tree
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.state_space import (
    causal_conv1d_varlen_fwd,
    gated_group_rmsnorm,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)

from ..model_config import NemotronHConfig
from .sharding import shard_dim0


def _layer_rows(rows: Tensor, layer: Tensor) -> TensorValue:
    """Row ``layer`` of a ``[num_layers, batch_size]`` table, as
    ``[1, batch_size]``.

    The row keeps its leading dim so the slice fuses into the kernels' slot
    input and is read in place rather than copied.
    """
    index = TensorValue(layer)
    return TensorValue(rows)[(slice(index, index + 1), 1), :]


def _causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    pool: Tensor,
    rows: Tensor,
    layer: Tensor,
    query_start_loc: Tensor,
    has_initial_state: Tensor,
) -> TensorValue:
    return causal_conv1d_varlen_fwd(
        x=TensorValue(x),
        weight=TensorValue(weight),
        bias=TensorValue(bias),
        conv_states=BufferValue(pool),
        query_start_loc=TensorValue(query_start_loc),
        cache_indices=_layer_rows(rows, layer),
        has_initial_state=TensorValue(has_initial_state),
        activation="silu",
        channels_last=True,
    )


def _ssd_scan(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dt_bias: Tensor,
    pool: Tensor,
    rows: Tensor,
    layer: Tensor,
    query_start_loc: Tensor,
    has_initial_state: Tensor,
) -> TensorValue:
    return mamba2_ssd_chunk_scan_varlen_fwd_inplace(
        x=TensorValue(x),
        dt=TensorValue(dt),
        A=TensorValue(A),
        B=TensorValue(B),
        C=TensorValue(C),
        D=TensorValue(D),
        dt_bias=TensorValue(dt_bias),
        ssm_pool=BufferValue(pool),
        query_start_loc=TensorValue(query_start_loc),
        has_initial_state=TensorValue(has_initial_state),
        cache_indices=_layer_rows(rows, layer),
    )


causal_conv1d = F.functional(_causal_conv1d)
ssd_scan = F.functional(_ssd_scan)


@tree.dataclass(frozen=True)
class MambaStateAccess:
    """One Mamba layer's view of the state pools.

    Each pool travels with the ``[num_layers, batch_size]`` pool rows of
    every layer, and the layer reads and writes the rows of ``layer``, one
    per request. A row sliced out before the shared subgraph would be copied
    to cross into it.
    """

    conv_pool: Tensor
    conv_rows: Tensor
    ssm_pool: Tensor
    ssm_rows: Tensor
    layer: Tensor
    """The int64 CPU scalar selecting this layer's rows."""


class CausalConv1d(
    Module[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
):
    """A depthwise causal conv with SiLU whose window lives in a state pool."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        self.weight = Tensor.zeros([dim, kernel_size])
        self.bias = Tensor.zeros([dim])

    def forward(
        self,
        x: Tensor,
        pool: Tensor,
        rows: Tensor,
        layer: Tensor,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
    ) -> Tensor:
        return causal_conv1d(
            x,
            self.weight,
            self.bias,
            pool,
            rows,
            layer,
            query_start_loc,
            has_initial_state,
        )


class GatedGroupRMSNorm(Module[[Tensor, Tensor], Tensor]):
    """``rms_norm(y * silu(gate))`` over each group of channels.

    HF's ``MambaRMSNormGated`` with ``norm_before_gate=False``.
    """

    def __init__(self, dim: int, group_size: int, eps: float) -> None:
        self.weight = Tensor.ones([dim])
        self.group_size = group_size
        self.eps = eps

    def forward(self, y: Tensor, gate: Tensor) -> Tensor:
        return F.functional(gated_group_rmsnorm)(
            y,
            gate,
            F.cast(self.weight, DType.float32),
            self.eps,
            self.group_size,
        )


class NemotronHMamba2Mixer(
    Module[[Tensor, MambaStateAccess, Tensor, Tensor], Tensor]
):
    """Mamba-2 selective state-space mixer.

    Both kernels read a request's state from its pool rows and write the new
    state back in place, so prefill, chunked prefill and decode are the same
    graph.
    """

    def __init__(self, config: NemotronHConfig) -> None:
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.n_groups = config.n_groups
        self.state_size = config.ssm_state_size
        self.intermediate = config.mamba_intermediate_size
        self.conv_dim = config.conv_dim

        # Each device runs its own heads and their groups.
        # permute_mamba_for_tp lays out the fused in_proj and conv rows so
        # that each device's share is one contiguous block.
        self.in_proj = ColumnParallelLinear(
            config.hidden_size,
            self.intermediate + self.conv_dim + self.num_heads,
            bias=False,
        )
        self.conv1d = CausalConv1d(self.conv_dim, config.conv_kernel)
        self.conv1d.weight = shard_dim0(self.conv1d.weight)
        self.conv1d.bias = shard_dim0(self.conv1d.bias)
        self.A_log = shard_dim0(Tensor.zeros([self.num_heads]))
        self.D = shard_dim0(Tensor.zeros([self.num_heads]))
        self.dt_bias = shard_dim0(Tensor.zeros([self.num_heads]))
        self.norm = GatedGroupRMSNorm(
            self.intermediate,
            self.intermediate // self.n_groups,
            config.layer_norm_epsilon,
        )
        self.norm.weight = shard_dim0(self.norm.weight)
        self.out_proj = RowParallelLinear(
            self.intermediate, config.hidden_size, bias=False
        )

    def forward(
        self,
        x: Tensor,
        state: MambaStateAccess,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
    ) -> Tensor:
        gate, xbc, dt = F.split(
            self.in_proj(x),
            [self.intermediate, self.conv_dim, self.num_heads],
            axis=1,
        )
        xbc = self.conv1d(
            xbc,
            state.conv_pool,
            state.conv_rows,
            state.layer,
            query_start_loc,
            has_initial_state,
        )
        # The conv, scan and norm kernels have no sharding rules. Each
        # device runs them on its own heads, so their outputs stay split on
        # the channel axis like their inputs.
        xbc = xbc.rebind_mapping(NamedMapping(xbc.mesh, (None, TP)))
        group_dim = self.n_groups * self.state_size
        hidden, B, C = F.split(
            xbc, [self.intermediate, group_dim, group_dim], axis=1
        )
        # The kernel applies softplus to dt + dt_bias itself.
        A = F.cast(F.negate(F.exp(F.cast(self.A_log, DType.float32))), x.dtype)
        y = ssd_scan(
            hidden.reshape([-1, self.num_heads, self.head_dim]),
            dt,
            A,
            B.reshape([-1, self.n_groups, self.state_size]),
            C.reshape([-1, self.n_groups, self.state_size]),
            self.D,
            self.dt_bias,
            state.ssm_pool,
            state.ssm_rows,
            state.layer,
            query_start_loc,
            has_initial_state,
        )
        y = y.rebind_mapping(NamedMapping(y.mesh, (None, TP, None)))
        y = self.norm(y.reshape([-1, self.intermediate]), gate)
        y = y.rebind_mapping(NamedMapping(y.mesh, (None, TP)))
        return self.out_proj(y)
