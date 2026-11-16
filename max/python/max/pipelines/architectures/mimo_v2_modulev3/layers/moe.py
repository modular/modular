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
"""MiMo-V2 routed experts with MXFP8 activations, on ModuleV3.

The routed experts run W4A8 -- MXFP4 weights, MXFP8 per-expert activations --
through :class:`max.nn._graph_ops.moe_quant.NvMxf4f8Strategy`; the modulev3 MoE
layers have no MXFP4/e8m0 path of their own. Each device holds its share of
every expert's channels, so its chain yields a partial sum the residual add
reduces.

The expert-parallel variant (:class:`MiMoV2ExpertParallelMoE`) holds whole
experts per device instead and moves the tokens through the EP
dispatch/combine kernels, quantized to the same MXFP8 activations in
transit.
"""

from __future__ import annotations

import dataclasses
import os

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferType, Graph, StaticDim, TensorValue, ops
from max.nn import Linear, Module
from max.nn._graph_ops import (
    MEGAFFN_MIMO_COUNTER_WORDS,
    MEGAFFN_MIMO_K1,
    MEGAFFN_MIMO_K2,
    MEGAFFN_MIMO_MAX_EXPERTS,
    MEGAFFN_MIMO_N1,
    _router_gate_mixed_gemv,
    megaffn_mimo_w4a8,
    moe_create_indices,
    moe_finalize,
)
from max.nn._graph_ops.ep import fused_silu
from max.nn._graph_ops.moe_quant import NvMxf4f8Strategy
from max.nn.activation import GatedActivation
from max.nn.common_layers import quant_ops
from max.nn.common_layers.mesh_axis import TP
from max.nn.common_layers.quant_tensor import FP8BlockTensor
from max.nn.kernels import moe_router_group_limited
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.quant_config import QuantConfig
from max.nn.sequential import ModuleList
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
    Sharded,
    match_operand_placement,
)

from ..quant import FP8_BLOCK

# The fused mixed router GEMV keeps the logits on the input's placement, which
# is replicated under tensor parallelism.
_router_gate_mixed_gemv_op = F.functional(
    _router_gate_mixed_gemv,
    rule=match_operand_placement(_router_gate_mixed_gemv, "hidden_states"),
)

MEGAFFN_MIMO_BAND_ROWS = 512
"""The most routed rows a step may carry and still take the one launch.

The band is ``MEGAFFN_MIMO_BAND_MIN_ROWS < rows <= 512``: a step inside it takes
the one launch, and a step outside it runs the two-launch chain. A decode step
routes 64 rows per request (top-8 over DFlash's 8 verify rows), so the ceiling
is a c8 step -- the widest one MiMo-V2.6-Flash serves.

It stays under the slot pool's limit. A single launch needs
``hidden_states.shape[0] // 8 + num_experts`` counters, and there are 4096 of
them: 512 rows over 256 experts is 320. A ceiling past that limit would admit
steps the counters cannot hold, and the op raises when the graph is built; this
one cannot. 0 keeps every step on the chain.
"""

MEGAFFN_MIMO_BAND_MIN_ROWS = 64
"""The fewest routed rows a step must carry and still take the one launch.

Exclusive: the one launch runs when ``64 < rows <= MEGAFFN_MIMO_BAND_ROWS``. The
floor keeps the shortest decode step on the chain, where the one launch still
loses -- 64 rows over 256 experts is a quarter row an expert on average -- and
c2's 128 rows are the shortest step that takes it. 0 lets every non-empty step
into the band.
"""


_MEGAFFN_MIMO_OPS = ("megaffn.mimo.w4a8",)
"""The one launch's op, registered only by the closed-source kernel packages
(``Kernels/lib/graph_compiler/mega_ffn_mimo.mojo``).
"""


def _mega_ffn_mimo_is_registered() -> bool:
    """Reports whether the current graph's kernel library has the one launch.

    A source-only build lacks the closed-source kernel packages that register
    it, so its experts run the grouped-matmul chain.
    """
    with ensure_context():
        library = Graph.current._kernel_library
        return all(op in library for op in _MEGAFFN_MIMO_OPS)


# The fused GEMV walks the gate weight's contraction in tiles of
# `simd_width * num_threads` elements with no tail guard, so a partly tiled K
# reads past the weight. The widest tile is SM100's 4-element vector load over
# 256 threads (1024), so a multiple of 1024 tiles every band.
_MIXED_GEMV_K_TILE = 1024


class _Fp8Weights(Module[[], None]):
    """One dense FP8 projection's weight and 128x128 block scales.

    Both stay on the host as stored; the graph gives each device its share.
    """

    def __init__(self, out_dim: int, in_dim: int) -> None:
        self.weight = Tensor.zeros(
            [out_dim, in_dim], dtype=DType.float8_e4m3fn, device=CPU()
        )
        self.weight_scale = Tensor.zeros(
            [out_dim // FP8_BLOCK, in_dim // FP8_BLOCK],
            dtype=DType.float32,
            device=CPU(),
        )

    def matmul(self, x: Tensor, axis: int, *, shard: bool = True) -> Tensor:
        """Returns ``x @ weight.T``, the weight sharded on ``axis``.

        A split on the weight's rows (``axis == 0``) is column-parallel and
        shards the output's columns; one on its columns leaves each device a
        partial sum of the output. With ``shard`` false the weight stays
        replicated and each device runs its own rows in full (data-parallel
        attention).
        """
        quant = FP8BlockTensor(
            weight=self.weight,
            weight_scale_inv=self.weight_scale,
            block_size=(FP8_BLOCK, FP8_BLOCK),
        )
        if shard:
            quant = quant.shard(axis, x.mesh)
        else:
            quant = quant.to(x.mesh)
        return quant_ops.matmul(x, quant)


class MiMoV2MLP(Module[[Tensor], Tensor]):
    """The dense FP8 SwiGLU MLP of the one non-MoE layer: ``down(silu(gate) * up)``."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        data_parallel: bool = False,
    ) -> None:
        self.data_parallel = data_parallel
        self.gate_proj = _Fp8Weights(intermediate_dim, hidden_dim)
        self.up_proj = _Fp8Weights(intermediate_dim, hidden_dim)
        self.down_proj = _Fp8Weights(hidden_dim, intermediate_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Runs the feed-forward over ``x``."""
        shard = not self.data_parallel
        gate = self.gate_proj.matmul(x, axis=0, shard=shard)
        up = self.up_proj.matmul(x, axis=0, shard=shard)
        return self.down_proj.matmul(F.silu(gate) * up, axis=1, shard=shard)


class MiMoV2Router(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Sigmoid top-k router whose logits are computed in float32.

    The reference scores with ``F.linear(hidden.float(), weight.float())``
    on the BF16 gate weight; the weight's cast runs once, at model init. The
    correction bias steers selection only; the combine weights are the
    unbiased sigmoid scores renormalized over the top k.

    On an NVIDIA GPU the gate runs as the fused mixed-input GEMV,
    which widens the BF16 activation to float32 in registers and dots it
    against that float32 weight in a single launch: the separate
    ``cast(hidden)`` kernel exists only to feed that GEMV, so decode row
    counts get the same logits from one launch instead of two. The op routes
    on the runtime row count: above 64 rows and up to its 128-row fused
    window on SM100 it keeps IEEE fp32 where the cast + matmul chain's fp32
    matmul has switched to TF32 tensor cores, so the two differ in the
    logits' last bits there (the fused value being the more precise), and
    past that window the op falls back to that same chain. The expert picks
    stay the same either way.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        route_on_every_device: bool = False,
    ) -> None:
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.route_on_every_device = route_on_every_device
        self.gate_score = Linear(hidden_dim, num_experts, bias=False)
        self.e_score_correction_bias = Tensor.zeros(
            [num_experts], dtype=DType.float32
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Routes ``x`` and returns the expert picks with their weights."""
        mesh = x.mesh
        if isinstance(x.placements[0], Sharded):
            # Data parallelism: each replica routes only its own rows, which
            # no other device holds, so there is nothing to agree on and no
            # device that could route for the others.
            return self._route(
                x, self.gate_score.weight, self.e_score_correction_bias
            )
        if self.route_on_every_device:
            # Bit-identical replicas of the router input: every device picks
            # the same experts without a collective.
            return self._route(
                x, self.gate_score.weight, self.e_score_correction_bias
            )
        # Fallback for a router whose owner did not vouch for rank-ordered
        # reductions: device 0 routes for every device. No MiMo-V2 mode lands
        # here, because the model flags both tensor-parallel modes and
        # data-parallel attention routes above.
        experts, weights = self._route(
            x.local_shards[0],
            self.gate_score.weight.local_shards[0],
            self.e_score_correction_bias.local_shards[0],
        )
        return (
            F.distributed_broadcast(experts, mesh),
            F.distributed_broadcast(weights, mesh),
        )

    def _gate_logits(self, x: Tensor, weight: Tensor) -> Tensor:
        """Returns the ``[tokens, num_experts]`` float32 gate logits."""
        if not self._fused_gate_applies(x, weight):
            return x.cast(DType.float32) @ weight.cast(DType.float32).T
        return _router_gate_mixed_gemv_op(x, weight.cast(DType.float32))

    def _fused_gate_applies(self, x: Tensor, weight: Tensor) -> bool:
        """Whether the fused GEMV can carry this call's gate.

        The op needs a BF16 activation and a statically known contraction it
        can walk in whole tiles, on a device its kernel package covers.
        Everything else keeps the cast + float32 matmul pair.

        On SM100 the op also prefetches its first weight tile before waiting
        on the dependent grid, which is only sound for a weight that is
        already initialized and unchanged for the whole execution. MiMo-V2's
        fp32 gate weight is an init-time constant: the compiler hoists the
        graph's cast of it into the init block (EICM), and the served
        engine's trace shows no per-call cast on the gate, so the prefetch is
        safe here.
        """
        k_dim = weight.shape[1]
        return (
            x.dtype == DType.bfloat16
            and isinstance(k_dim, StaticDim)
            and int(k_dim) % _MIXED_GEMV_K_TILE == 0
            and all(device.api == "cuda" for device in x.mesh.devices)
        )

    def _route(
        self, x: Tensor, weight: Tensor, bias: Tensor
    ) -> tuple[Tensor, Tensor]:
        logits = self._gate_logits(x, weight)
        scores = F.sigmoid(logits)
        return moe_router_group_limited(
            scores,
            bias,
            self.num_experts,
            self.num_experts_per_tok,
            n_groups=1,
            topk_group=1,
            norm_weights=self.norm_topk_prob,
            routed_scaling_factor=1.0,
        )


class _MxWeights(Module[[], None]):
    """One routed expert's MXFP4 projection, as the checkpoint stores it.

    Packed E2M1 codes with one E8M0 scale per 32 inputs; the stacks stay on
    the host, and the graph cuts each device's share.
    """

    def __init__(self, out_dim: int, in_dim: int) -> None:
        self.weight = Tensor.zeros(
            [out_dim, in_dim // 2], dtype=DType.uint8, device=CPU()
        )
        self.weight_scale = Tensor.zeros(
            [out_dim, in_dim // 32],
            dtype=DType.float8_e8m0fnu,
            device=CPU(),
        )


class MiMoV2Expert(Module[[], None]):
    """One routed expert's gate, up and down projections."""

    def __init__(self, moe_dim: int, hidden_dim: int) -> None:
        self.gate_proj = _MxWeights(moe_dim, hidden_dim)
        self.up_proj = _MxWeights(moe_dim, hidden_dim)
        self.down_proj = _MxWeights(hidden_dim, moe_dim)


class MiMoV2MoE(Module[..., Tensor]):
    """MiMo-V2's sigmoid-routed MoE, with no shared expert.

    On NVIDIA, the experts run W4A8: each projection's input is quantized to
    MXFP8 per expert at runtime. The gate/up matmul applies the gated
    activation and the down projection's re-quantize in its epilogue. Every
    device runs every routed row through its share of each expert's channels,
    which gives a partial sum of the output the residual add reduces.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_dim: int,
        norm_topk_prob: bool,
        quant_config: QuantConfig,
        route_on_every_device: bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_dim = moe_dim
        self.activation = GatedActivation()
        # The chained activation writes bfloat16, so the fused epilogue rounds
        # its SwiGLU output through the same bfloat16 before it quantizes.
        self.fused_activation = dataclasses.replace(
            self.activation, round_to_bf16=True
        )
        self.quant_config = quant_config
        # MAX_DISABLE_FUSED_SWIGLU_NVFP4 is the escape hatch the graph-API MoE
        # reads through QuantConfig.can_use_fused_swiglu, which a W4A8
        # checkpoint leaves unset. Unlike that flag, this one keeps the
        # sigma-permuted stacks and runs the chained kernels on them, with
        # fused_silu(interleaved=True), so one weight layout serves both arms.
        self.fused_swiglu = (
            os.environ.get("MAX_DISABLE_FUSED_SWIGLU_NVFP4") != "1"
        )
        self.gate = MiMoV2Router(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            route_on_every_device=route_on_every_device,
        )
        self.experts = ModuleList(
            [MiMoV2Expert(moe_dim, hidden_dim) for _ in range(num_experts)]
        )
        # MXFP4 inputs give about 5x the expert error of MXFP8 against a
        # float32 reference (test_expert_precision_gpu.py), so the experts'
        # activations stay MXFP8.
        self.strategy = NvMxf4f8Strategy(quant_config, DType.float8_e4m3fn)

    def _stacked_weights(
        self, mesh: DeviceMesh
    ) -> list[tuple[TensorValue, ...]]:
        """Each device's gate/up and down stacks, with interleaved scales.

        Reads only weights, so the graph compiler runs it once, at model
        init. The gate/up stacks interleave each expert's gate rows with its
        up rows, the pairing the fused gate/up grouped matmul reads.
        """
        e, md, hidden = self.num_experts, self.moe_dim, self.hidden_dim
        n = mesh.num_devices
        # Each device's share of every expert's channels: rows of the gate
        # and up projections, columns of the down projection. Each weight
        # stack is built whole on the host, whose parallel concat reads the
        # checkpoint in order, and each device then receives only its final
        # share: a device temporary freed between the persistent weight
        # stacks strands allocator granules the KV cache cannot use. The
        # scale interleave's device temporaries are small enough to ignore.
        # TODO(DISTINF-813): F.stack(device=...) would place the shares
        # directly, but its gather_scatter.shard_and_stack copies each device's slice
        # out of the checkpoint mapping serially (TP4 init 118 s -> 264 s
        # from a cold page cache). Return to it once DISTINF-813 is fixed.
        # The private helper is the N-axis permutation
        # `grouped_matmul_blocked_swiglu` requires (gate row i beside up row
        # i), unnamed publicly because its wrapper takes a quant tensor. It
        # copies on the host; a device's slice stays contiguous.
        rows = NamedMapping(mesh, (None, TP, None))
        cols = NamedMapping(mesh, (None, None, TP))
        gate_up_stack = quant_ops._sigma_permute_rows(
            F.stack(
                [
                    w
                    for x in self.experts
                    for w in (x.gate_proj.weight, x.up_proj.weight)
                ],
                axis=0,
            ).reshape([e, 2 * md, hidden // 2])
        ).to(rows)
        gate_up_scale_stack = quant_ops._sigma_permute_rows(
            F.stack(
                [
                    w
                    for x in self.experts
                    for w in (x.gate_proj.weight_scale, x.up_proj.weight_scale)
                ],
                axis=0,
            ).reshape([e, 2 * md, hidden // 32])
        ).to(rows)
        down = F.stack([x.down_proj.weight for x in self.experts], axis=0).to(
            cols
        )
        down_scale = F.stack(
            [x.down_proj.weight_scale for x in self.experts], axis=0
        ).to(cols)
        per_device: list[tuple[TensorValue, ...]] = []
        for d in range(n):
            gate_up = TensorValue(gate_up_stack.local_shards[d]).reshape(
                [e, 2 * md // n, hidden // 2]
            )
            gate_up_scales = TensorValue(
                gate_up_scale_stack.local_shards[d]
            ).reshape([e, 2 * md // n, hidden // 32])
            down_w = TensorValue(down.local_shards[d])
            gate_up_interleaved, down_interleaved = (
                self.strategy.prepare_weight_scales(
                    gate_up_scales,
                    TensorValue(down_scale.local_shards[d]),
                    down_w.device,
                )
            )
            per_device.append(
                (gate_up, gate_up_interleaved, down_w, down_interleaved)
            )
        return per_device

    def _experts_on_device(
        self,
        x: TensorValue,
        router_idx: TensorValue,
        router_weight: TensorValue,
        gate_up: TensorValue,
        gate_up_scales: TensorValue,
        down: TensorValue,
        down_scales: TensorValue,
    ) -> TensorValue:
        """Runs the W4A8 expert FFN on one device."""
        strategy = self.strategy
        group = self.quant_config.input_scale.block_size
        assert group is not None
        (
            token_order,
            expert_start,
            restore_order,
            expert_ids,
            usage_stats,
            scales_offset,
        ) = moe_create_indices(
            ops.cast(ops.reshape(router_idx, [-1]), DType.int32),
            self.num_experts,
            needs_scales_offset=True,
        )
        total_m = ops.shape_to_tensor(token_order.shape)[0].cast(DType.uint32)
        # The grouped quantize gathers rows through the slot order itself,
        # dividing by the top-k in its loads, so neither the token indices
        # nor the permuted activations materialize.
        permuted_quant, permuted_scales = strategy.grouped_quantize(
            x,
            group[1],
            None,
            expert_start,
            scales_offset,
            expert_ids,
            indices=token_order,
            indices_divisor=self.num_experts_per_tok,
        )

        def chained() -> TensorValue:
            """The two-launch chain: the fused gate/up, then the down matmul."""
            # The fused gate/up matmul applies the gated activation and its
            # MXFP8 re-quantize in the epilogue: two launches where the
            # chained path needs four.
            expert_inputs = (
                permuted_quant,
                permuted_scales,
                expert_start,
                scales_offset,
                expert_ids,
                usage_stats,
            )
            if self.fused_swiglu:
                gate_up_quant, gate_up_quant_scales = (
                    strategy.grouped_matmul_swiglu(
                        gate_up,
                        gate_up_scales,
                        expert_inputs=expert_inputs,
                        estimated_total_m=total_m,
                        activation=self.fused_activation,
                    )
                )
            else:
                gate_up_out = strategy.grouped_matmul(
                    gate_up,
                    gate_up_scales,
                    expert_inputs=expert_inputs,
                    estimated_total_m=total_m,
                )
                # The interleaved weights put (gate, up) on adjacent output
                # columns, which the activation reads as pairs.
                gate_up_out = fused_silu(
                    gate_up_out,
                    expert_start,
                    activation=self.activation,
                    interleaved=True,
                )
                gate_up_quant, gate_up_quant_scales = strategy.grouped_quantize(
                    gate_up_out,
                    group[1],
                    None,
                    expert_start,
                    scales_offset,
                    expert_ids,
                )
            return strategy.grouped_matmul(
                down,
                down_scales,
                expert_inputs=(
                    gate_up_quant,
                    gate_up_quant_scales,
                    expert_start,
                    scales_offset,
                    expert_ids,
                    usage_stats,
                ),
                estimated_total_m=total_m,
            )

        if not self._mega_ffn_applies(x, permuted_quant, gate_up, down):
            return moe_finalize(
                chained(), restore_order, router_weight, x.dtype
            )

        # The counters the one launch stages its per-pool arrivals in: one
        # buffer a device, not a layer, because every MoE layer calls one
        # shared compiled body (`as_subgraph(self.mlp, name="mimo_v2_moe")`)
        # and this staged buffer belongs to that body. Sharing is safe because
        # a device runs one launch at a time on its stream, so a layer's
        # arrivals drain before the next layer's launch begins; two launches
        # overlapping on one buffer would interleave the halves of a slot word
        # and pin the walk's generation parity. A separate compilation of the
        # model -- a base MEF and a speculative one, say -- stages its own.
        # Persistent, not per-call state: allocated and zeroed when the model
        # loads, then each launch counts into one half of every slot word and
        # clears the half its predecessor used, so the parity survives a
        # capture and replay.
        counter = ops.buffer_create(
            BufferType(DType.uint32, [MEGAFFN_MIMO_COUNTER_WORDS], x.device),
            init_value=0,
        )

        # The regime is the op's own choice, made on the host from the routed
        # rows: inside `floor < rows <= band` it runs the one launch, which is
        # bit-equal to the chain, and outside it the chain's own kernels, which
        # is what the graph would have run there anyway. Nothing outside the op
        # branches, so the eager prefill path has no `mo.if` region to record
        # and poll events around -- 47 of them a step, each a shape the engine
        # cannot launch straight through. The count is a host-side shape value,
        # so a captured decode step records the one arm its batch takes.
        #
        # Two bounds, not one: the one launch loses to the chain at the shortest
        # decode step as well as at prefill-sized ones, so only the range its
        # own bench covers takes it.
        down_out, _packed, _scales = megaffn_mimo_w4a8(
            permuted_quant,
            gate_up,
            down,
            permuted_scales,
            gate_up_scales,
            down_scales,
            expert_start,
            scales_offset,
            expert_ids,
            None,
            None,
            counter,
            band_rows=MEGAFFN_MIMO_BAND_ROWS,
            floor_rows=MEGAFFN_MIMO_BAND_MIN_ROWS,
            activation=self.fused_activation,
        )
        return moe_finalize(down_out, restore_order, router_weight, x.dtype)

    def _mega_ffn_applies(
        self,
        x: TensorValue,
        permuted_quant: TensorValue,
        gate_up: TensorValue,
        down: TensorValue,
    ) -> bool:
        """Whether the one-launch MegaFFN can carry this call's experts.

        The op is ``Kernels/lib/mega_ffn/mega_ffn_mimo``'s W4A8 kernel, locked
        to MiMo-V2.6-Flash's per-rank expert slice: 4096 hidden over 1024
        gate-up rows a device, then 4096 down rows over a 512-wide hidden, i.e.
        four-way tensor parallelism. Any other split -- the tile the kernel's
        MMA shape is built on moves with it -- keeps the chain, as does the
        unfused-chain switch. The one launch applies the
        activation it is handed, but ``silu_bf16`` is the only one it was
        measured and byte-gated with, so any other activation, a clamp, or a
        graph-op activation in its place keeps the chain too. MiMo's config
        refuses a device that is not an SM100 GPU, which is the only
        architecture the kernel builds for, and a build without the closed
        kernel packages has no such op: emitting one there would fail the graph
        on an unknown op instead of running the chain it ran before.
        """
        if not _mega_ffn_mimo_is_registered():
            return False
        # The op runs the fused gate/up chain out of band whatever this says, so
        # the unfused-chain switch has to keep the op out of the graph as well:
        # otherwise setting it alone would still run the fused epilogue here and
        # the escape hatch would not mean what it says.
        if not self.fused_swiglu:
            return False
        # A graph-op activation is a function the kernel does not compute, and
        # one no kernel can fuse.
        if self.fused_activation.gated_activation_fn is not None:
            return False
        parameters = self.fused_activation.kernel_parameters()
        # The op applies whatever activation it is handed in both arms, but
        # the one launch was measured and byte-gated only with plain
        # silu(gate) * up rounded through bfloat16, the model's activation.
        # Another name (swigluoai, situ, gelu_tanh), or an unrounded, clamped
        # or scaled silu, keeps the chain until it has been gated too.
        if (parameters["activation"], parameters["swiglu_limit"]) != (
            "silu_bf16",
            0.0,
        ):
            return False
        if parameters["swiglu_alpha"] != 0.0:
            return False
        if permuted_quant.dtype != DType.float8_e4m3fn:
            return False
        # The walk scans a fixed window of experts, which the op checks by
        # raising when the graph is built. A config past it keeps the chain
        # rather than turning an optimization gate into a failed graph.
        if self.num_experts > MEGAFFN_MIMO_MAX_EXPERTS:
            return False
        hidden, gate_up_rows, down_k = (
            x.shape[1],
            gate_up.shape[1],
            down.shape[2],
        )
        if not (
            isinstance(hidden, StaticDim)
            and isinstance(gate_up_rows, StaticDim)
            and isinstance(down_k, StaticDim)
            and isinstance(gate_up.shape[0], StaticDim)
        ):
            return False
        return (int(hidden), int(gate_up_rows), int(down_k) * 2) == (
            MEGAFFN_MIMO_K1,
            MEGAFFN_MIMO_N1,
            MEGAFFN_MIMO_K2,
        ) and int(gate_up.shape[0]) == self.num_experts

    def forward(self, x: Tensor) -> Tensor:
        """Routes ``x`` through its experts and combines their outputs."""
        experts, weights = self.gate(x)
        mesh = x.mesh
        n = mesh.num_devices
        stacked = self._stacked_weights(mesh)
        with ensure_context():
            shards = [
                self._experts_on_device(
                    TensorValue(x.local_shards[d]),
                    TensorValue(experts.local_shards[d]),
                    TensorValue(weights.local_shards[d]),
                    *stacked[d],
                )
                for d in range(n)
            ]
        if n == 1:
            return Tensor.from_shard_values(shards, x.mapping)
        return Tensor.from_shard_values(
            shards, DeviceMapping(mesh, (Partial(),))
        )


class MiMoV2ExpertParallelMoE(MiMoV2MoE):
    """MiMo-V2's MoE with whole experts sharded across the devices.

    Each device owns one contiguous block of ``num_experts // n_devices``
    whole experts -- full ``moe_dim`` width, no channel split -- and the EP
    dispatch/combine kernels move the tokens between router and experts.
    Which backend serves that depends on the residual stream's layout: with
    the replicated stream of tensor-parallel attention the allreduce backend
    routes every device's token replica to its local experts and the
    combine returns that device's share of the top-8 sum, which the
    residual add's allreduce completes. With data-parallel attention each
    token lives on one replica, so the P2P backend moves each replica's rows
    to the global expert set and the combine returns their complete sums,
    with no residual collective at all.

    The W4A8 chain is the same one :class:`MiMoV2MoE` runs: the dispatch
    kernel quantizes the tokens to MXFP8 (E4M3 data, E8M0 32-block scales,
    the NVIDIA block-scaled layout) as it sends them, so the local grouped
    matmuls consume the dispatched activations directly -- no separate
    activation quantize on the receiving device. It keeps the chained
    gate/up, activation and re-quantize kernels rather than
    :class:`MiMoV2MoE`'s fused epilogue, which no EP arm here measured;
    folding this class into the shared one (DISTINF-675) brings it along.

    Hand-rolled rather than the stock ModuleV3
    :class:`~max.nn.common_layers.quant_moe.ExpertParallelMoE`,
    which a W4A8 model cannot use: its experts must be ``QuantizedMLP`` s
    backed by ``FP8BlockTensor``/``NVFP4Tensor`` (no MXFP4 quant tensor
    exists), and ``EPDispatchPayload.from_dispatch`` keys its column
    parsing on the routed-expert *weight* format, so an MXFP4 config takes
    the plain-FP8 arm and misreads the 6-column block-scaled dispatch.
    TODO(DISTINF-675): key ``EPDispatchPayload`` on the *dispatch* format
    and add an MXFP4 quant tensor when a second W4A8 architecture needs
    ModuleV3 EP, then fold this class back into the shared one.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_dim: int,
        norm_topk_prob: bool,
        quant_config: QuantConfig,
        ep_batch_manager: EPBatchManager,
        route_on_every_device: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_dim=moe_dim,
            norm_topk_prob=norm_topk_prob,
            quant_config=quant_config,
            route_on_every_device=route_on_every_device,
        )
        self.ep_batch_manager = ep_batch_manager

    def _stacked_weights(
        self, mesh: DeviceMesh
    ) -> list[tuple[TensorValue, ...]]:
        """Each device's whole local experts' gate/up and down stacks.

        Device ``d`` owns experts ``[d * local, (d + 1) * local)`` at their
        full width. Like the tensor-parallel stacks this reads only weights,
        so the graph compiler runs it once, at model init.
        """
        n = mesh.num_devices
        if self.num_experts % n:
            raise ValueError(
                f"MiMo-V2: {n} devices do not divide the "
                f"{self.num_experts} routed experts."
            )
        local = self.num_experts // n
        md, hidden = self.moe_dim, self.hidden_dim
        block = list(self.experts)
        per_device: list[tuple[TensorValue, ...]] = []
        for d in range(n):
            experts = block[d * local : (d + 1) * local]
            device = mesh.devices[d]
            # One [gate; up] block per expert, the layout fused_silu's
            # split mode reads, stacked on the host so that only the final
            # stack reaches the device (see the tensor-parallel stacks).
            gate_up = TensorValue(
                F.stack(
                    [
                        w
                        for x in experts
                        for w in (x.gate_proj.weight, x.up_proj.weight)
                    ],
                    axis=0,
                ).to(device)
            ).reshape([local, 2 * md, hidden // 2])
            gate_up_scales = TensorValue(
                F.stack(
                    [
                        w
                        for x in experts
                        for w in (
                            x.gate_proj.weight_scale,
                            x.up_proj.weight_scale,
                        )
                    ],
                    axis=0,
                ).to(device)
            ).reshape([local, 2 * md, hidden // 32])
            down_w = TensorValue(
                F.stack([x.down_proj.weight for x in experts], axis=0).to(
                    device
                )
            )
            down_scale = F.stack(
                [x.down_proj.weight_scale for x in experts], axis=0
            ).to(device)
            gate_up_interleaved, down_interleaved = (
                self.strategy.prepare_weight_scales(
                    gate_up_scales,
                    TensorValue(down_scale),
                    down_w.device,
                )
            )
            per_device.append(
                (gate_up, gate_up_interleaved, down_w, down_interleaved)
            )
        return per_device

    def _local_experts(
        self,
        dispatched: tuple[TensorValue, ...],
        stacked: tuple[TensorValue, ...],
        estimated_total_m: TensorValue,
    ) -> TensorValue:
        """Runs the W4A8 expert chain on one device's dispatched tokens.

        The dispatch hands back the received rows already grouped by local
        expert, in the NVIDIA block-scaled layout: MXFP8 data with E8M0
        scales in per-expert padded tiles, plus the grouped-matmul metadata.
        """
        gate_up, gate_up_scales, down, down_scales = stacked
        (
            tokens,
            token_scales,
            expert_start,
            scales_offset,
            expert_ids,
            usage_stats,
        ) = dispatched
        strategy = self.strategy
        group = self.quant_config.input_scale.block_size
        assert group is not None
        gate_up_out = strategy.grouped_matmul(
            gate_up,
            gate_up_scales,
            expert_inputs=(
                tokens,
                token_scales,
                expert_start,
                scales_offset,
                expert_ids,
                usage_stats,
            ),
            estimated_total_m=estimated_total_m,
        )
        gate_up_out = fused_silu(
            gate_up_out,
            expert_start,
            activation=self.activation,
            interleaved=False,
        )
        gate_up_quant, gate_up_scales = strategy.grouped_quantize(
            gate_up_out, group[1], None, expert_start, scales_offset, expert_ids
        )
        return strategy.grouped_matmul(
            down,
            down_scales,
            expert_inputs=(
                gate_up_quant,
                gate_up_scales,
                expert_start,
                scales_offset,
                expert_ids,
                usage_stats,
            ),
            estimated_total_m=estimated_total_m,
        )

    def forward(self, x: Tensor, comm: EPCommBuffers | None = None) -> Tensor:
        """Runs the expert-parallel forward pass.

        It gates, dispatches to the local experts, then combines their
        outputs.

        Args:
            x: The residual stream, ``[tokens, hidden]`` -- replicated
                under tensor-parallel attention, sharded per replica under
                data-parallel attention.
            comm: The graph's EP communication buffers. Optional only to
                keep the base ``forward(self, x)`` signature compatible; the
                EP path always supplies it.

        Returns:
            The layer's partial MoE output, one shard per device, which the
            residual add's allreduce reduces under the allreduce backend;
            each replica's complete output in place under data parallelism.
        """
        batch_mgr = self.ep_batch_manager
        config = batch_mgr.config
        # Data-parallel attention shards the batch over the replicas, so
        # each token lives on exactly one device -- the P2P dispatch's
        # precondition. The replicated (TP) residual stream needs the
        # allreduce backend instead: its dispatch routes only the local
        # experts and the residual allreduce completes the partial
        # combines, which cannot cross replicas holding different rows.
        dp_attention = isinstance(x.placements[0], Sharded)
        if config.use_allreduce and dp_attention:
            raise NotImplementedError(
                "MiMo-V2 data-parallel attention needs the P2P expert "
                "backend (--ep-use-allreduce off); the allreduce backend "
                "routes each device's tokens to its local experts only and "
                "relies on the residual allreduce to complete the top-k "
                "sum, which cannot cross data-parallel replicas holding "
                "different rows."
            )
        if not config.use_allreduce and not dp_attention:
            # The P2P dispatch assumes each token lives on one device
            # (``tp_sharded`` residual); MiMo's is replicated, so every
            # replica of a token would be dispatched and the combine would
            # double-count it. Serving that needs the reduce-scatter /
            # allgather residual restructure.
            raise NotImplementedError(
                "MiMo-V2 expert parallelism runs on the allreduce backend"
                " only; the P2P backend needs a sequence-sharded residual"
                " stream."
            )
        assert comm is not None, (
            "MiMoV2ExpertParallelMoE.forward requires comm buffers"
        )
        batch_mgr.bind_comm_buffers(comm)
        mesh = x.mesh
        n = mesh.num_devices
        device_ids = [d.id for d in mesh.devices]
        experts, weights = self.gate(x)
        stacked = self._stacked_weights(mesh)
        # Under tensor parallelism every device holds the same rows and the
        # same routing decision: bit-identical replicas, or device 0's
        # broadcast.
        x_shards = [x.local_shards[d] for d in range(n)]
        # The cast runs at the graph level: the router's output placement is
        # unknown inside the shared subgraph, which the functional cast's
        # planner refuses.
        idx_shards = [
            TensorValue(shard).cast(DType.int32)
            for shard in experts.local_shards
        ]
        w_shards = [weights.local_shards[d] for d in range(n)]
        # The allreduce backend routes each device's own replica to its
        # local experts; no token crosses devices. `ep_dispatch`/`ep_combine`
        # take one constrained type for every tensor argument, so each tensor
        # crosses the boundary as its graph value, like the TP path's calls.
        dispatched: list[tuple[TensorValue, ...]] = [
            tuple(
                TensorValue(value)
                for value in batch_mgr.ep_dispatch(
                    TensorValue(x_shards[d]), idx_shards[d], device_ids[d]
                )
            )
            for d in range(n)
        ]
        # The estimate wants every row a device receives for its local
        # experts, about tokens * top_k / n_gpus. Under TP the residual is
        # replicated, so one shard's count is the whole batch; under DP the
        # shards hold disjoint replicas whose P2P dispatches all land on
        # this device's experts, so they sum.
        if dp_attention:
            tokens = F.shape_to_tensor(x_shards[0].shape)[0]
            for shard in x_shards[1:]:
                tokens = tokens + F.shape_to_tensor(shard.shape)[0]
        else:
            tokens = F.shape_to_tensor(x_shards[0].shape)[0]
        estimated_total_m = TensorValue(
            (tokens * self.num_experts_per_tok // config.n_gpus_per_node).cast(
                DType.uint32
            )
        )
        with ensure_context():
            downs = [
                Tensor.from_graph_value(
                    self._local_experts(
                        dispatched[d], stacked[d], estimated_total_m
                    )
                )
                for d in range(n)
            ]
        combines = [
            batch_mgr.ep_combine(
                TensorValue(downs[d]),
                TensorValue(w_shards[d]),
                device_ids[d],
                idx_shards[d],
            )
            for d in range(n)
        ]
        if dp_attention:
            # P2P: each replica's rows come back from the experts as their
            # complete top-k sums, in place; the plain residual add needs no
            # collective.
            return Tensor.from_shard_values(combines, x.mapping)
        return Tensor.from_shard_values(
            combines, DeviceMapping(mesh, (Partial(),))
        )
