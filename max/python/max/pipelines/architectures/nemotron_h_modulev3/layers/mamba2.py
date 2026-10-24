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

from collections.abc import Sequence
from typing import Generic, TypeVar

from max import tree
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh, NamedMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.state_space import (
    causal_conv1d_varlen_fwd,
    gated_group_rmsnorm,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)

from ..model_config import NemotronHConfig
from .quantized import quantized_linear
from .sharding import shard_axis, shard_segments


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
    update_state: Tensor,
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
        update_state=TensorValue(update_state),
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
    update_state: Tensor,
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
        update_state=TensorValue(update_state),
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


_ReplayTensor = TypeVar("_ReplayTensor", Tensor, TensorValue)


@tree.dataclass(frozen=True)
class MambaReplayInputs(Generic[_ReplayTensor]):
    """What a replay of one Mamba layer's state updates reads.

    The verify's own activations and the two per-head weights of the scan,
    never a full SSM state per verified token. The pre-verify state remains
    in the live Jenga leaves.
    """

    projected: _ReplayTensor
    """The verify's rows of the ``in_proj`` output: the conv input and ``dt``."""
    convolved: _ReplayTensor
    """The verify's rows of the conv output: the scan's ``x`` and ``B``."""
    A: _ReplayTensor
    dt_bias: _ReplayTensor


@tree.dataclass(frozen=True)
class MambaReplayCapture:
    """The verify's rows of the two tensors a Mamba layer's replay reads.

    Every per-token replay input is a column slice of one of them, so a verify
    copies two tensors per layer.
    """

    projected: Tensor
    """The captured rows of the ``in_proj`` output."""
    convolved: Tensor
    """The captured rows of the convolution's output."""


class CausalConv1d(
    Module[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
):
    """A depthwise causal conv with SiLU whose window lives in a state pool.

    Each device convolves its own channels.
    """

    def __init__(
        self, dim: int, kernel_size: int, segments: Sequence[int] = ()
    ) -> None:
        """Initializes the conv.

        Args:
            dim: The number of channels.
            kernel_size: The window.
            segments: The sizes of the parts the channels stack, each split
                across devices, or none to split the channels as one part.
        """
        self.segments = tuple(segments)
        if self.segments:
            # A plain split would give a device whole parts. The weights
            # stay on the host, for the graph to cut each device's share.
            self.weight = Tensor.zeros([dim, kernel_size], device=CPU())
            self.bias = Tensor.zeros([dim], device=CPU())
        else:
            self.weight = shard_axis(Tensor.zeros([dim, kernel_size]))
            self.bias = shard_axis(Tensor.zeros([dim]))

    def placed(self, mesh: DeviceMesh) -> tuple[Tensor, Tensor]:
        """Returns the weight and bias, each device holding its channels."""
        if not self.segments:
            return self.weight, self.bias
        return (
            shard_segments(self.weight, self.segments, mesh),
            shard_segments(self.bias, self.segments, mesh),
        )

    def forward(
        self,
        x: Tensor,
        pool: Tensor,
        rows: Tensor,
        layer: Tensor,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
        update_state: Tensor,
    ) -> Tensor:
        weight, bias = self.placed(x.mesh)
        return causal_conv1d(
            x,
            weight,
            bias,
            pool,
            rows,
            layer,
            query_start_loc,
            has_initial_state,
            update_state,
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
    Module[[Tensor, MambaStateAccess, Tensor, Tensor, Tensor], Tensor]
):
    """Mamba-2 selective state-space mixer.

    Both kernels read a request's state from its pool rows and write the new
    state back in place, so prefill, chunked prefill and decode are the same
    graph. A request whose ``update_state`` flag is false only reads its rows,
    which lets a speculative verify run on the live state and leave it as it
    was.
    """

    def __init__(self, config: NemotronHConfig, name: str) -> None:
        """Initializes the mixer.

        Args:
            config: The model config.
            name: The mixer's checkpoint path, which names the format of
                its projections.
        """
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.n_groups = config.n_groups
        self.state_size = config.ssm_state_size
        self.intermediate = config.mamba_intermediate_size
        self.conv_dim = config.conv_dim

        # Each device runs its own heads and their groups, so it takes its
        # share of every part the fused in_proj and conv rows stack.
        group_dim = self.n_groups * self.state_size
        conv_parts = (self.intermediate, group_dim, group_dim)
        self.in_proj = quantized_linear(
            config,
            f"{name}.in_proj",
            row_segments=(self.intermediate, *conv_parts, self.num_heads),
        )
        self.conv1d = CausalConv1d(
            self.conv_dim,
            config.conv_kernel,
            conv_parts if len(config.devices) > 1 else (),
        )
        self.A_log = shard_axis(Tensor.zeros([self.num_heads]))
        self.D = shard_axis(Tensor.zeros([self.num_heads]))
        self.dt_bias = shard_axis(Tensor.zeros([self.num_heads]))
        self.norm = GatedGroupRMSNorm(
            self.intermediate,
            self.intermediate // self.n_groups,
            config.layer_norm_epsilon,
        )
        self.norm.weight = shard_axis(self.norm.weight)
        self.out_proj = quantized_linear(config, f"{name}.out_proj")

    def forward(
        self,
        x: Tensor,
        state: MambaStateAccess,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
        update_state: Tensor,
    ) -> Tensor:
        output, _ = self.forward_with_replay(
            x, state, query_start_loc, has_initial_state, update_state
        )
        return output

    def _decay(self, dtype: DType) -> Tensor:
        """The per-head ``A``, ``-exp(A_log)``, in the activation dtype.

        The verify and its replay must read the same ``A``, so both take it
        from here.
        """
        return F.cast(F.negate(F.exp(F.cast(self.A_log, DType.float32))), dtype)

    def replay_statics(self, dtype: DType) -> tuple[Tensor, Tensor]:
        """Returns the layer weights a replay of its state updates reads.

        The conv window is the accepted inputs themselves, so only the scan
        reads weights.

        Args:
            dtype: The activation dtype, which ``A`` is cast to.

        Returns:
            ``(A, dt_bias)``, as the verify's scan reads them.
        """
        return self._decay(dtype), self.dt_bias

    def forward_with_replay(
        self,
        x: Tensor,
        state: MambaStateAccess,
        query_start_loc: Tensor,
        has_initial_state: Tensor,
        update_state: Tensor,
        capture_rows: Tensor | None = None,
    ) -> tuple[Tensor, MambaReplayCapture | None]:
        """Returns the mixer output and the inputs needed to replay its state.

        Args:
            x: The normalized layer input, ``[tokens, hidden]``.
            state: This layer's view of the state pools.
            query_start_loc: The ragged offsets of the sequences.
            has_initial_state: Whether each sequence resumes from its rows.
            update_state: Whether each sequence writes its new state back to
                its rows.
            capture_rows: The ``[captured]`` token rows to return the replay
                inputs of, or ``None`` to return none.
        """
        projected = self.in_proj(x)
        gate, xbc, dt = F.split(
            projected,
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
            update_state,
        )
        # The conv, scan and norm kernels have no sharding rules. Each
        # device runs them on its own heads, so their outputs stay split on
        # the channel axis like their inputs.
        convolved = xbc.rebind_mapping(NamedMapping(xbc.mesh, (None, TP)))
        group_dim = self.n_groups * self.state_size
        hidden, B, C = F.split(
            convolved, [self.intermediate, group_dim, group_dim], axis=1
        )
        # The kernel applies softplus to dt + dt_bias itself.
        A = self._decay(x.dtype)
        hidden = hidden.reshape([-1, self.num_heads, self.head_dim])
        B = B.reshape([-1, self.n_groups, self.state_size])
        C = C.reshape([-1, self.n_groups, self.state_size])
        y = ssd_scan(
            hidden,
            dt,
            A,
            B,
            C,
            self.D,
            self.dt_bias,
            state.ssm_pool,
            state.ssm_rows,
            state.layer,
            query_start_loc,
            has_initial_state,
            update_state,
        )
        y = y.rebind_mapping(NamedMapping(y.mesh, (None, TP, None)))
        y = self.norm(y.reshape([-1, self.intermediate]), gate)
        y = y.rebind_mapping(NamedMapping(y.mesh, (None, TP)))
        replay: MambaReplayCapture | None = None
        if capture_rows is not None:
            # Only the captured rows survive the layer, not every token's.
            replay = MambaReplayCapture(
                projected=F.gather(projected, capture_rows, axis=0),
                convolved=F.gather(convolved, capture_rows, axis=0),
            )
        return self.out_proj(y), replay
