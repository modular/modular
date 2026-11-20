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
"""Mistral Large 4 mixture of experts: 128 routed experts plus one shared.

The routed experts are NVFP4 (W4A4, modelopt encoding) and go through the
grouped NVFP4 chain; the shared expert is FP8 block-scaled and is added on the
same post-attention-norm input. The router is Mistral's fp32 softmax top-4 (see
:mod:`.moe_gate`), so this reuses the grouped-matmul call the DeepSeek-V3.2 MoE
makes but none of its sigmoid/noaux router.

Three implementations share that router: the replicated one for a single
device or pure data parallelism, :class:`MistralLarge4TensorParallelMoE`,
which splits every expert's FFN width across the ranks, and
:class:`MistralLarge4ExpertParallelMoE`, which places ``128 / n_devices``
experts per rank and moves tokens through the expert-parallel dispatch and
combine.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Module
from max.nn.activation import GatedActivation
from max.nn.common_layers import quant_ops
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_moe import (
    _SHARED_EXPERT_OVERLAP_ENV,
    ExpertParallelMoE,
    QuantizedMLP,
    QuantizedMoE,
    TensorParallelMoE,
    _shared_expert_overlap,
    _stack_experts,
)
from max.nn.common_layers.quant_tensor import NVFP4Tensor, QuantAwareTensor
from max.nn.kernels import moe_finalize
from max.nn.legacy.quant_config import QuantConfig
from max.nn.sequential import ModuleList
from max.tensor import Tensor, default_device, default_dtype, defaults
from max.tensor.sharding import DeviceMapping, NamedMapping, Partial

from .mega_moe_tp import MAX_FUSED_TOKENS, mega_moe_tp, mega_moe_tp_enabled
from .mlp import MistralLarge4MLP
from .moe_gate import MistralLarge4Router

_OVERLAP_SHARED_EXPERT_ENV = _SHARED_EXPERT_OVERLAP_ENV
"""Opts the shared expert onto a side stream, for A/B measurement only.

``1`` puts the shared expert on a side stream and ``0`` leaves it inline; unset
leaves each layout on its own default, which for both of this model's layouts is
inline (see
:meth:`MistralLarge4ExpertParallelMoE._compute_shared_experts` for the
expert-parallel layer's reason and
:meth:`MistralLarge4TensorParallelMoE.forward` for the tensor-parallel one's).
Anything else is an error, so the variable cannot mean "on" to one layout and
"off" to another.

``1`` arms *both* layouts, including the expert-parallel region that stays off
by default because its output is not reproducible, so set it per export arm:
in a shell that also stages the served expert-parallel export, it re-enables
that region. It is read where the graph is built, so it has to be set for the
export: an exported MEF carries whichever arm it was built with and a Mach boot
cannot switch it.
"""


class MistralLarge4MoE(Module[..., Tensor]):
    """Sparse MoE with a shared expert, routed through grouped matmuls."""

    gate: MistralLarge4Router
    experts: ModuleList[MistralLarge4MLP]
    shared_experts: MistralLarge4MLP

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        expert_quant_config: QuantConfig | None,
        shared_expert_quant_config: QuantConfig | None,
    ) -> None:
        """Builds the router, the routed experts and the shared expert.

        Args:
            hidden_dim: Width of the hidden state.
            num_experts: Routed experts (128).
            num_experts_per_token: Experts a token takes (4).
            moe_dim: Hidden width of one expert.
            expert_quant_config: The routed experts' NVFP4 config.
            shared_expert_quant_config: The shared expert's FP8 block config.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_dim = moe_dim
        self.quant_config = expert_quant_config
        self.gate_up_accum_dtype = None
        self.combine_accum_dtype = None
        self.swiglu_limit = 0.0
        # Plain SiLU on the routed experts: ``None`` is what the grouped chain
        # reads off a module that skips ``QuantizedMoE.__init__``.
        self.activation: GatedActivation | None = None
        self.gate = MistralLarge4Router(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.experts = ModuleList(
            [
                MistralLarge4MLP(
                    hidden_dim, moe_dim, quant_config=expert_quant_config
                )
                for _ in range(num_experts)
            ]
        )
        self.shared_experts = MistralLarge4MLP(
            hidden_dim, moe_dim, quant_config=shared_expert_quant_config
        )
        _, self.mesh = defaults()

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """The stacked ``[gate, up]`` expert weight bundle."""
        per_expert: list[QuantAwareTensor] = [
            quant_ops.concat_weights(expert.w1.weight, expert.w3.weight, axis=0)
            for expert in self.experts
        ]
        return [_stack_experts(per_expert, shard_axis=None, mesh=self.mesh)]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """The stacked down-projection expert weight bundle."""
        per_expert: list[QuantAwareTensor] = [
            expert.w2.weight for expert in self.experts
        ]
        return [_stack_experts(per_expert, shard_axis=None, mesh=self.mesh)]

    def _combine_expert_outputs(
        self,
        down_projs: Tensor,
        restore_token_order: Tensor,
        router_weight: Tensor,
        dtype: DType,
    ) -> Tensor:
        """Restores token order and weight-combines the per-token expert outputs.

        ``QuantizedMoE._routed_experts`` is reused unbound with this module as
        ``self``, so it needs the same combine hook the base class carries.
        """
        return moe_finalize(
            down_projs, restore_token_order, router_weight, dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """Routes ``x`` through the experts and adds the shared expert."""
        router_idx, router_weight = self.gate(x)
        routed = F.call_on_mesh(
            QuantizedMoE._routed_experts,
            x.mesh,
            out_specs=router_weight.mapping,
        )(
            self,
            x,
            router_idx,
            router_weight,
            self.gate_up_proj[0],
            self.down_proj[0],
        )
        return routed + self.shared_experts(x)


class MistralLarge4ExpertParallelMoE(ExpertParallelMoE):
    """Mistral Large 4 MoE with its 128 routed experts spread over the ranks.

    ``ExpertParallelMoE`` owns the dispatch/compute/combine sequence and the
    per-device expert stacking, but it reaches for DeepSeek's module layout:
    experts named ``gate_proj``/``up_proj``/``down_proj`` and one quant config
    for both the routed and the shared experts. Mistral Large 4 keeps the
    checkpoint's ``w1``/``w3``/``w2`` names (the weight contract and therefore
    the Mach loader fix them) and gives the shared expert its own FP8 block
    quantization, so those three hooks are re-expressed here and the inherited
    :meth:`forward` is used unchanged. It also runs the shared expert inline,
    for the reason :meth:`_compute_shared_experts` gives.

    Args:
        hidden_dim: Width of the hidden state.
        num_experts: Global routed-expert count (128).
        num_experts_per_token: Experts a token takes (4).
        moe_dim: Hidden width of one expert.
        expert_quant_config: The routed experts' NVFP4 config.
        shared_expert_quant_config: The shared expert's FP8 block config.
        ep_batch_manager: The expert-parallel runtime this layer dispatches on.
    """

    experts: ModuleList[QuantizedMLP]
    shared_experts: MistralLarge4MLP

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        expert_quant_config: QuantConfig | None,
        shared_expert_quant_config: QuantConfig | None,
        ep_batch_manager: Any,
    ) -> None:
        # Deliberately skips ``QuantizedMoE.__init__``: it would build
        # DeepSeek's gate and hand the shared expert the routed experts' quant
        # config, both of which this checkpoint contradicts.
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_dim = moe_dim
        self.quant_config = expert_quant_config
        self.swiglu_limit = 0.0
        self.activation: GatedActivation | None = None
        self.ep_batch_manager = ep_batch_manager
        self.gate = MistralLarge4Router(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.shared_experts = MistralLarge4MLP(
            hidden_dim, moe_dim, quant_config=shared_expert_quant_config
        )
        _, self.mesh = defaults()
        self.experts = self._init_experts()

    def _init_experts(self) -> ModuleList[QuantizedMLP]:
        """Builds this rank's share of the routed experts."""
        dtype, mesh = defaults()
        del dtype
        if mesh.ndim != 1:
            raise ValueError(
                "Expert-parallel Mistral Large 4 needs a one-axis device mesh,"
                f" got {mesh}"
            )
        if self.num_experts % mesh.num_devices != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must divide the device"
                f" count ({mesh.num_devices}) for expert parallelism"
            )
        local = self.num_experts // mesh.num_devices
        experts: list[QuantizedMLP] = []
        for device in mesh.devices:
            with default_device(device):
                experts.extend(
                    MistralLarge4MLP(
                        self.hidden_dim,
                        self.moe_dim,
                        quant_config=self.quant_config,
                    )
                    for _ in range(local)
                )
        return ModuleList[QuantizedMLP](experts)

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device stacked ``[gate, up]`` bundle of the local experts."""
        per_device: list[list[list[QuantAwareTensor]]] = [
            [] for _ in self.mesh.devices
        ]
        local = self.num_experts // self.mesh.num_devices
        for n, expert in enumerate(self.experts):
            assert isinstance(expert, MistralLarge4MLP)
            per_device[n // local].append(
                [
                    quant_ops.concat_weights(
                        expert.w1.weight, expert.w3.weight, axis=0
                    )
                ]
            )
        stacked = self._stack_local_experts(per_device)
        if self._uses_fused_swiglu:
            permuted: list[QuantAwareTensor] = []
            for w in stacked:
                assert isinstance(w, NVFP4Tensor)
                permuted.append(quant_ops.sigma_permute_gate_up_nvfp4(w))
            return permuted
        return stacked

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device stacked down-projection bundle of the local experts."""
        per_device: list[list[list[QuantAwareTensor]]] = [
            [] for _ in self.mesh.devices
        ]
        local = self.num_experts // self.mesh.num_devices
        for n, expert in enumerate(self.experts):
            assert isinstance(expert, MistralLarge4MLP)
            per_device[n // local].append([expert.w2.weight])
        return self._stack_local_experts(per_device)

    def _nvfp4_global_input_scale(self) -> Tensor:
        """The scalar max static gate ``input_scale`` across local experts."""
        scales: list[Tensor] = []
        for expert in self.experts:
            assert isinstance(expert, MistralLarge4MLP)
            weight = expert.w1.weight
            assert isinstance(weight, NVFP4Tensor)
            scales.append(weight.input_scale)
        return F.max(F.stack(scales, axis=0), axis=0)

    def _compute_shared_experts(
        self, x: Tensor, devices: Sequence[int]
    ) -> list[Tensor]:
        """Runs the shared expert on the main stream for ``devices``.

        The base class puts the shared expert on a side stream by default, so
        it overlaps the routed experts' communication. In this model's graphs
        the region's output is not reproducible: interleaving the two arms run
        by run inside one process diverged, and a greedy request does not
        repeat identically across boots with the overlap on.

        TODO(GEX-4482): run the shared expert on the base class's side stream
        again once the graph compiler synchronizes that region, and delete this
        override with it.

        ``MODULAR_OVERLAP_SHARED_EXPERT=1`` selects the overlap when the graph
        is built, which exists only to measure what it costs; ``0`` leaves it
        inline and unset is the same as ``0`` here. Note that the same value
        arms :meth:`MistralLarge4TensorParallelMoE.forward`, so set it for the
        arm being staged rather than globally.

        Args:
            x: The hidden state, replicated or distributed over the mesh.
            devices: The mesh positions to run, in order.

        Returns:
            One shared-expert output per entry of ``devices``, in order.
        """
        if _shared_expert_overlap(default=False):
            return super()._compute_shared_experts(x, devices)
        return self._shared_experts_inline(x, devices)


def _tensor_parallel_mlp(layer: MistralLarge4MLP) -> MistralLarge4MLP:
    """Shards ``w1``/``w3`` on their output axis and ``w2`` on its input axis.

    The same weight placements
    :func:`~max.nn.common_layers.quant_linear.tensor_parallel_mlp` makes,
    spelled for the checkpoint's projection names and applied to the weights
    directly. ``col_parallel``/``row_parallel`` would land them too, and would
    also shard a bias, which this layer does not have (every projection is
    built with ``bias=False``); they are typed for a float ``Tensor`` weight
    and cannot take a quantized one without a suppression, whereas the dense
    layer 0 FFN in :mod:`.transformer_block` places its weights this way for
    the same reason.

    Args:
        layer: The layer to shard, mutated in place.

    Returns:
        The same layer, for use in an expression.
    """
    for projection in (layer.w1, layer.w3):
        projection.weight = projection.weight.to(
            NamedMapping(projection.weight.mesh, (TP, None))
        )
    layer.w2.weight = layer.w2.weight.to(
        NamedMapping(layer.w2.weight.mesh, (None, TP))
    )
    return layer


class MistralLarge4TensorParallelMoE(TensorParallelMoE):
    """Mistral Large 4 MoE with every expert's FFN width split over the ranks.

    ``TensorParallelMoE`` holds a ``1 / n_devices`` slice of every expert
    rather than a subset of the experts, so no token leaves its rank: the
    routed experts and the shared expert each return a partial sum, which the
    block's ``post_mlp`` turns into one all-reduce. It reaches for DeepSeek's
    module layout (experts named ``gate_proj``/``up_proj``/``down_proj``, built
    by ``QuantizedMoE.__init__`` off one quant config), so the expert builder
    and the two weight-bundle hooks are re-expressed here for the checkpoint's
    ``w1``/``w3``/``w2`` names and the shared expert's own FP8 block config,
    the same way :class:`MistralLarge4ExpertParallelMoE` re-expresses them.

    The gate/up bundle stays in checkpoint order rather than taking the
    expert-parallel bundle's fused-SwiGLU permutation: that kernel is selected
    only for pre-quantized dispatch activations
    (:func:`~max.nn.common_layers.quant_ops.grouped_matmul_silu`), so a rank
    here runs the plain grouped-matmul chain.

    :meth:`forward` runs the shared expert on the main stream, which is the
    order this model's graphs are reproducible in (see
    :meth:`MistralLarge4ExpertParallelMoE._compute_shared_experts`), and puts it
    on a side stream only when ``MODULAR_OVERLAP_SHARED_EXPERT=1`` asks for it.
    The routed chain here is per-device compute with no communication to hide
    under, so what the side stream buys is the region's GEMMs and quantizes
    overlapping that chain: measured at TP8 on B200 against the unfused routed
    chain, about 1.3 ms of a 13.0 ms base-graph decode step and 0.4 ms of a
    5.3 ms DSpark verify step.

    With ``MODULAR_MEGA_MOE_TP=1`` set when the graph is built, the routed
    experts run as one ``mega_moe.tp`` op (see :mod:`.mega_moe_tp`): a
    decode or verify step as one fused launch, a larger step as the same
    chain inside the op, bit-identical either way. The op reads the gate/up
    bundle sigma-permuted, so the flag permutes it for every graph.

    The two flags compose for correctness, not for speed. The overlap forks
    the shared expert ahead of the routed segment, which it reaches through
    ``TensorParallelMoE._routed_experts_out`` and so through
    :meth:`_routed_experts`, and with both set that segment is the one
    ``mega_moe.tp`` op. The fused launch keeps a CTA resident on every SM for
    its grid collectives, so it cannot share the device with the region: the
    two run one after the other, about the region's time plus the fused
    launch's per block. The region's kernels can delay the launch but never
    block it, as none of them waits on it. The combination's benefit is
    unmeasured and probably negative; the 1.3 ms above does not carry over.

    Args:
        hidden_dim: Width of the hidden state.
        num_experts: Routed experts (128).
        num_experts_per_token: Experts a token takes (4).
        moe_dim: Hidden width of one expert, before the split.
        expert_quant_config: The routed experts' NVFP4 config.
        shared_expert_quant_config: The shared expert's FP8 block config.
        graph_mode: The graph's mode; a ``"prefill"`` graph never takes the
            fused launch. The combined graph (``"auto"``) chooses by the
            step's token count alone, so a prefill step of at most
            ``MAX_FUSED_TOKENS`` tokens (a short prompt, a chunk's tail) takes
            it too. That is deliberate: the two regimes are bit-identical, and
            telling a prefill step from a decode step inside a captured graph
            would take a host-side flag into the op for a choice of speed
            alone.
    """

    experts: ModuleList[QuantizedMLP]
    shared_experts: MistralLarge4MLP

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        expert_quant_config: QuantConfig | None,
        shared_expert_quant_config: QuantConfig | None,
        graph_mode: str = "auto",
    ) -> None:
        # Deliberately skips ``TensorParallelMoE.__init__``, which would build
        # DeepSeek's gate and hand the shared expert the routed experts' quant
        # config; the sequence below is the rest of it.
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_dim = moe_dim
        self._initial_moe_dim = moe_dim
        self.quant_config = expert_quant_config
        self.swiglu_limit = 0.0
        self.activation: GatedActivation | None = None
        self._mega_moe_tp = (
            mega_moe_tp_enabled()
            and expert_quant_config is not None
            and expert_quant_config.is_nvfp4
            and num_experts_per_token == 4
        )
        self._max_fused_tokens = (
            0 if graph_mode == "prefill" else MAX_FUSED_TOKENS
        )
        self.gate = MistralLarge4Router(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.shared_experts = MistralLarge4MLP(
            hidden_dim, moe_dim, quant_config=shared_expert_quant_config
        )
        _, self.mesh = defaults()
        self.experts = self._init_experts()
        self._set_mesh(self.mesh)
        self.shared_experts = _tensor_parallel_mlp(self.shared_experts)
        for n, expert in enumerate(self.experts):
            assert isinstance(expert, MistralLarge4MLP)
            self.experts[n] = _tensor_parallel_mlp(expert)

    def _init_experts(self) -> ModuleList[QuantizedMLP]:
        """Builds every routed expert whole, on the host off a mesh."""
        dtype, mesh = defaults()
        placement = (
            contextlib.nullcontext()
            if mesh.num_devices == 1
            else default_device(CPU())
        )
        with placement, default_dtype(dtype):
            return ModuleList[QuantizedMLP](
                [
                    MistralLarge4MLP(
                        self.hidden_dim,
                        self.moe_dim,
                        quant_config=self.quant_config,
                    )
                    for _ in range(self.num_experts)
                ]
            )

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device ``[w1, w3]`` weight bundle, sharded along ``moe_dim``."""
        interleaved: list[QuantAwareTensor] = []
        gates: list[QuantAwareTensor] = []
        for expert in self.experts:
            assert isinstance(expert, MistralLarge4MLP)
            interleaved.extend((expert.w1.weight, expert.w3.weight))
            gates.append(expert.w1.weight)

        def _combine_gate_up(tensors: list[Tensor]) -> list[Tensor]:
            # The shard is along axis 0, so each leaf's trailing dim is
            # preserved: ``hidden_dim`` for the data leaf, ``hidden_dim //
            # block_k`` for an FP8 block-scale leaf.
            return self._shard_stack_tensors(
                tensors,
                axis=0,
                shard_shape=[self.num_experts, -1, tensors[0].shape[-1]],
            )

        # ``interleaved`` has two entries per expert, but an NVFP4 expert has a
        # single global scale pair shared by its two projections, so those come
        # off the gate weights.
        stacked = quant_ops.combine_quant_per_device(
            interleaved, _combine_gate_up, global_scale_items=gates
        )
        if not self._mega_moe_tp:
            return stacked
        # Each rank's [w1 | w3] slice, gate row i moved to 2i and up row i to
        # 2i + 1: the layout the fused op's SwiGLU epilogue pairs rows in.
        permuted: list[QuantAwareTensor] = []
        for w in stacked:
            assert isinstance(w, NVFP4Tensor)
            permuted.append(quant_ops.sigma_permute_gate_up_nvfp4(w))
        return permuted

    def forward(self, x: Tensor) -> Tensor:
        """Runs the routed experts and adds the shared expert.

        Unset or ``0``, this is ``TensorParallelMoE.forward`` unchanged: the
        shared expert sits on the main stream, after the routed chain. With
        ``MODULAR_OVERLAP_SHARED_EXPERT=1`` its body is emitted on stream 1
        *before* the routed chain instead, so the two run concurrent and only
        the add joins them. Both sides are per-rank partial sums over the TP
        axis, so the block's ``post_mlp`` all-reduce still covers them once.
        The routed chain is re-expressed here rather than inherited because the
        base's ``forward`` does the add itself, which leaves no point outside it
        at which to fork the shared expert ahead of the chain -- but the chain
        itself is the base's own :meth:`_routed_experts_out`, so the two arms
        cannot diverge.

        Args:
            x: ``(seq_len, hidden_dim)``, replicated over the TP axis: each rank
                holds the full hidden state and the weights are sharded.

        Returns:
            ``(seq_len, hidden_dim)``, each device's partial sum.
        """
        if not _shared_expert_overlap(default=False):
            return super().forward(x)

        router_idx, router_weight = self.gate(x)
        # Mesh positions, not device ids: ``local_shards`` is positional.
        shared_shards = self._compute_shared_experts(
            x, range(self.mesh.num_devices)
        )
        routed_expert_out = self._routed_experts_out(
            x, router_idx, router_weight
        )

        shared_out = Tensor.from_shard_values(
            list(shared_shards), DeviceMapping(self.mesh, (Partial(),))
        )
        if self.combine_accum_dtype is not None:
            shared_out = shared_out.cast(self.combine_accum_dtype)
        routed_expert_out += shared_out
        if self.combine_accum_dtype is not None:
            routed_expert_out = routed_expert_out.cast(x.dtype)
        return routed_expert_out

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device ``w2`` bundle, sharded along ``moe_dim``."""
        down_list: list[QuantAwareTensor] = []
        for expert in self.experts:
            assert isinstance(expert, MistralLarge4MLP)
            down_list.append(expert.w2.weight)

        distributed = _stack_experts(down_list, shard_axis=-1, mesh=self.mesh)
        return list(distributed.local_shards)

    def _routed_experts(
        self,
        x: Tensor,
        router_idx: Tensor,
        router_weight: Tensor,
        gate_up: QuantAwareTensor,
        down: QuantAwareTensor,
    ) -> Tensor:
        """Routes one rank's tokens through its slice of every expert.

        Without ``MODULAR_MEGA_MOE_TP`` this is the base class's chain. With
        it, the whole chain from the gather to the finalize is one
        ``mega_moe.tp`` op on the sigma-permuted gate/up bundle, which picks
        the fused launch or the chain from the step's token count.
        """
        if not self._mega_moe_tp:
            return super()._routed_experts(
                x, router_idx, router_weight, gate_up, down
            )
        assert isinstance(gate_up, NVFP4Tensor) and isinstance(
            down, NVFP4Tensor
        )
        if router_weight.dtype != DType.float32:
            raise ValueError(
                "mega_moe.tp takes float32 router weights, got"
                f" {router_weight.dtype}"
            )
        # The op lays out the routing ``moe_create_indices`` would from the
        # top-k ids itself, and its scalars are the ones the unfused chain's
        # grouped matmul and quantize derive, so the two arms read identical
        # values.
        return mega_moe_tp(
            x,
            F.cast(F.reshape(router_idx, [-1]), DType.int32),
            router_weight,
            gate_up.weight,
            quant_ops.interleave_grouped_scales(gate_up.weight_scale),
            down.weight,
            quant_ops.interleave_grouped_scales(down.weight_scale),
            (gate_up.weight_scale_2 * gate_up.input_scale).to(x.mesh),
            (down.weight_scale_2 * down.input_scale).to(x.mesh),
            (1.0 / gate_up.input_scale).to(x.mesh),
            (1.0 / down.input_scale).to(x.mesh),
            max_fused_tokens=self._max_fused_tokens,
        )
