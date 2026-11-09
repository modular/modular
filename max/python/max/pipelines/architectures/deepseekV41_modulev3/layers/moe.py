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
"""V4.1 local routed experts and the tensor-parallel shared expert.

The routed experts are NVFP4 or MXFP4, as the checkpoint's
:class:`~..model_config.RoutedExpertFormat` says.
"""

from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.graph import TensorValue
from max.nn import Module
from max.nn.activation import GatedActivation
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_linear import QuantizedLinear
from max.nn.common_layers.quant_ops import (
    EPDispatchPayload,
    ep_requires_dispatch_scales,
    sigma_permute_gate_up,
)
from max.nn.common_layers.quant_tensor import (
    MXFP4Tensor,
    NVFP4Activation,
    NVFP4Tensor,
    QuantAwareTensor,
)
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.quant_config import QuantConfig
from max.nn.sequential import ModuleList
from max.nn.transparent_module import TransparentModule
from max.tensor import Tensor, default_device, defaults
from max.tensor.sharding import DeviceMapping, DeviceMesh, Partial
from max.tensor.sharding.action import PerShard

from ..model_config import RoutedExpertFormat
from . import mxfp4_ops, nvfp4_ops
from .mxfp8_linear import MXFP8Linear, Mxfp8Rows


class _Expert(Module[[], None]):
    w1: QuantizedLinear
    w3: QuantizedLinear
    w2: QuantizedLinear

    def __init__(
        self, hidden_dim: int, moe_dim: int, expert_format: RoutedExpertFormat
    ) -> None:
        super().__init__()
        quant_config = expert_format.weight_quant_config()
        self.w1 = QuantizedLinear(
            hidden_dim, moe_dim, bias=False, quant_config=quant_config
        )
        self.w3 = QuantizedLinear(
            hidden_dim, moe_dim, bias=False, quant_config=quant_config
        )
        self.w2 = QuantizedLinear(
            moe_dim, hidden_dim, bias=False, quant_config=quant_config
        )

    def forward(self) -> None:
        raise NotImplementedError("Experts are evaluated as local groups")


class RoutedExperts(TransparentModule[[Tensor, Tensor, Tensor], Tensor]):
    """Local quantized experts followed by float32 all-reduce.

    NVFP4 experts run W4A4 against the checkpoint's static activation
    scales; MXFP4 experts run W4A8, quantizing each GEMM's input to MXFP8
    as the reference does. Under expert parallelism (:meth:`ep_forward`)
    each rank runs its local experts on what it received through the P2P
    dispatch, and the combine returns every rank its own rows complete.

    Drops its parent attribute name so leaves stay ``ffn.experts.E.wN``.

    Args:
        hidden_dim: Token width.
        moe_dim: Expert intermediate width.
        num_experts: Global expert count (384 for the full checkpoint).
        mesh: One-dimensional tensor-parallel mesh; defaults to the context.
        swiglu_limit: Asymmetric clamp limit from training.
        ep_batch_manager: The EP batch manager when the routed experts run
            expert-parallel; ``None`` keeps the tensor-parallel path.
        expert_format: How the checkpoint quantizes the experts.
    """

    experts: ModuleList[_Expert]

    def __init__(
        self,
        hidden_dim: int,
        moe_dim: int,
        num_experts: int = 384,
        *,
        mesh: DeviceMesh | None = None,
        swiglu_limit: float | None = 10.0,
        ep_batch_manager: EPBatchManager | None = None,
        expert_format: RoutedExpertFormat = RoutedExpertFormat.NVFP4,
    ) -> None:
        super().__init__()
        self.mesh = mesh or defaults()[1]
        self.expert_format = expert_format
        self.hidden_dim = hidden_dim
        self.moe_dim = moe_dim
        self.ep_batch_manager = ep_batch_manager
        if self.mesh.ndim != 1 or num_experts % self.mesh.num_devices:
            raise ValueError("Experts must divide evenly across a 1-D mesh")
        if hidden_dim % 32 or moe_dim % 32:
            raise ValueError("Expert widths must be divisible by 32")
        self.num_experts = num_experts
        self.per_rank = num_experts // self.mesh.num_devices
        self.swiglu_limit = swiglu_limit
        experts = []
        for e in range(num_experts):
            device = self.mesh.devices[e // self.per_rank]
            with default_device(device):
                expert = _Expert(hidden_dim, moe_dim, expert_format)
            # NVFP4Tensor.zeros puts the scalars on the host. The grouped
            # quantize kernel reads them on the device every step, so a host
            # copy would be freed under it; place them at load time instead.
            for linear in (expert.w1, expert.w3, expert.w2):
                tensor = linear.weight
                if isinstance(tensor, NVFP4Tensor):
                    tensor.weight_scale_2 = tensor.weight_scale_2.to(device)
                    tensor.input_scale = tensor.input_scale.to(device)
            experts.append(expert)
        self.experts = ModuleList(experts)
        if expert_format is RoutedExpertFormat.NVFP4:
            # The dispatch quantizes every rank's activations with one
            # uniform scale: the max static ``w1``/``w3`` ``input_scale``
            # over all experts, a pure function of the checkpoint the weight
            # adapter materializes as this leaf. Reading it here replaces
            # the in-graph cross-device max the EP forward used to run per
            # step -- two 4-byte scalar hubs whose gather/scatter lower to
            # device-wait pairs. MXFP8 dispatch scales each block by its
            # own max, so MXFP4 experts have no such leaf.
            self.global_input_scale = Tensor.zeros(
                [1], dtype=DType.float32, device=self.mesh
            )

    def forward(
        self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> Tensor:
        """Combines global int32 ids and normalized, pre-scaled f32 weights."""
        return F.allreduce_sum(self.partial(x, topk_ids, topk_weights))

    def _local_bundles(
        self,
    ) -> list[tuple[QuantAwareTensor, QuantAwareTensor, QuantAwareTensor]]:
        """Per-rank stacked local weights: ``(w1, w3, w2)``.

        Each stack holds its rank's ``per_rank`` experts in expert order,
        the layout the grouped matmuls and the EP local compute read.
        """
        bundles = []
        for rank in range(self.mesh.num_devices):
            local = self.experts[
                rank * self.per_rank : (rank + 1) * self.per_rank
            ]
            bundles.append(
                (
                    quant_ops.stack([e.w1.weight for e in local]),
                    quant_ops.stack([e.w3.weight for e in local]),
                    quant_ops.stack([e.w2.weight for e in local]),
                )
            )
        return bundles

    def ep_forward(
        self,
        x: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        comm: EPCommBuffers,
        total_rows: Tensor,
    ) -> Tensor:
        """The EP forward over per-rank rows: dispatch, compute, combine.

        Every rank routes its own rows (the gate weight is replicated),
        dispatches each (token, expert) pair to the rank owning the expert
        through the P2P EP buffers, quantized to the experts' activation
        format (NVFP4 against one uniform scale, or MXFP8), runs its local
        experts on what it received, and combines the outputs back to the
        owning rank with the router weights applied. The local experts run
        as the fused gate/up + clamped SwiGLU + re-quantize grouped matmul
        followed by the down projection's grouped matmul, which the graph
        compiler fuses into one MegaFFN launch. The result is each rank's
        own rows, complete: bf16 in ``x``'s placement.

        ``comm`` must be the buffers bound from the EP graph inputs; see
        :meth:`~max.nn.legacy.comm.ep.EPBatchManager.bind_comm_buffers`.

        ``total_rows`` is the batch's total token rows as a host-resident
        scalar (the sum of every rank's shard rows, which the trunk reads
        off the CPU ``input_row_offsets_i64`` input). The grouped kernels'
        tile-regime estimate derives from it on the host, so the launch
        stays free of both the cross-device row-count sum and the
        per-consuming-op device-to-host materialization the in-graph
        estimate used to cost.
        """
        mgr = self.ep_batch_manager
        if mgr is None:
            raise ValueError("ep_forward requires an EP batch manager")
        if topk_ids.dtype != DType.int32 or topk_weights.dtype != DType.float32:
            raise ValueError("Routing requires int32 ids and float32 weights")
        mgr.bind_comm_buffers(comm)
        config = mgr.config
        quant_config = config.dispatch_quant_config
        if quant_config is None or not quant_config.can_use_fused_swiglu:
            raise ValueError(
                "the V4.1 EP path requires a quantized dispatch the fused "
                "SwiGLU kernel reads"
            )

        x_shards = list(x.local_shards)
        ids_shards = list(topk_ids.local_shards)
        weight_shards = list(topk_weights.local_shards)
        device_ids = [d.id for d in self.mesh.devices]

        global_scales: list[Tensor] | None = None
        input_scales: list[Tensor] | None = None
        if ep_requires_dispatch_scales(quant_config):
            # One replicated [1] leaf; ``local_shards`` is per rank, in mesh
            # order, exactly where the dispatch and the activation payloads
            # read their scale.
            global_scales = list(self.global_input_scale.local_shards)
            input_scales = [
                F.broadcast_to(scale, [self.num_experts])
                for scale in global_scales
            ]

        dispatch_results = mgr.ep_dispatch_all(
            x_shards, ids_shards, device_ids, input_scales=input_scales
        )
        payload = EPDispatchPayload.from_dispatch(
            dispatch_results, quant_config, config
        )
        assert payload.scales is not None and payload.scales_offset is not None
        tokens: Sequence[QuantAwareTensor]
        if global_scales is None:
            tokens = payload.per_device_tokens()
        else:
            # The NVFP4 branch of ``per_device_tokens``, with each device's
            # activation carrying its own copy of the uniform scale (the
            # shared helper takes one scale for every device).
            tokens = [
                NVFP4Activation(
                    data=data,
                    scales=scale,
                    input_scale=global_scales[i],
                    scales_offset=offset,
                )
                for i, (data, scale, offset) in enumerate(
                    zip(
                        payload.tokens,
                        payload.scales,
                        payload.scales_offset,
                        strict=True,
                    )
                )
            ]

        # Estimated token-expert pairs across all ranks, for the grouped
        # kernels' tile regime. The total row count arrives from the host,
        # so this stays host arithmetic: the estimate feeds the grouped
        # composites as a launch operand they read in place, with no
        # device round-trip to materialize it.
        estimated_total_m = (
            total_rows * int(topk_ids.shape[1]) // config.n_gpus_per_node
        ).cast(DType.uint32)

        down_bundle = [
            self.dispatched(
                rank,
                tokens[rank],
                payload.expert_start[rank],
                payload.expert_ids[rank],
                payload.usage_stats[rank]
                if payload.usage_stats is not None
                else None,
                payload.scales_offset[rank],
                quant_config,
                estimated_total_m,
            )
            for rank in range(self.mesh.num_devices)
        ]
        combine_results = mgr.ep_combine_all(
            down_bundle, weight_shards, device_ids
        )
        out = Tensor.from_shard_values(
            [TensorValue(shard) for shard in combine_results], x.mapping
        )
        return out.cast(DType.bfloat16)

    def dispatched(
        self,
        rank: int,
        tokens: QuantAwareTensor,
        expert_start: Tensor,
        expert_ids: Tensor,
        usage_stats: Tensor | None,
        scales_offset: Tensor,
        quant_config: QuantConfig,
        estimated_total_m: Tensor,
    ) -> Tensor:
        """Runs ``rank``'s experts over the rows its dispatch delivered.

        The gate/up grouped matmul runs with the clamped SwiGLU and the
        down input's re-quantize in its epilogue, and the graph compiler
        fuses it with the down projection's grouped matmul into one MegaFFN
        launch. Both read only the received rows.

        Args:
            rank: The mesh rank whose local experts run.
            tokens: The received rows, quantized in the dispatch format and
                sorted by local expert.
            expert_start: The ``[groups + 1]`` uint32 row offsets.
            expert_ids: The ``[groups]`` int32 local expert of each group.
            usage_stats: The dispatch's host usage stats, if it has them.
            scales_offset: Each group's scale-tile offset.
            quant_config: The dispatch's quantization.
            estimated_total_m: Estimated rows received per device, on host.

        Returns:
            ``[rows, hidden]`` bf16 down projections, without routing
            weights: the combine applies them after the down projection.
            The single-device path (:func:`.mxfp4_ops.routed`) instead
            scales the SwiGLU output before the down input's MXFP8
            quantize, as the reference does, so the two round differently.
        """
        # The fused gate/up kernel reads adjacent output columns as
        # (gate, up) pairs: stack each expert's ``[w1; w3]`` rows, then
        # sigma-permute (the ExpertParallelMoE weight layout).
        local = self.experts[rank * self.per_rank : (rank + 1) * self.per_rank]
        stacked = quant_ops.stack(
            [
                quant_ops.concat_weights(
                    expert.w1.weight, expert.w3.weight, axis=0
                )
                for expert in local
            ]
        )
        assert isinstance(stacked, (NVFP4Tensor, MXFP4Tensor))
        gate_up = sigma_permute_gate_up(stacked)
        down = quant_ops.stack([expert.w2.weight for expert in local])
        down_in = quant_ops.grouped_matmul_silu(
            tokens,
            gate_up,
            down,
            expert_start,
            expert_ids,
            usage_stats,
            quant_config,
            scales_offset=scales_offset,
            estimated_total_m=estimated_total_m,
            activation=GatedActivation("silu", limit=self.swiglu_limit or 0.0),
        )
        return quant_ops.grouped_matmul(
            down_in,
            down,
            expert_start,
            expert_ids,
            usage_stats,
            scales_offset=scales_offset,
            estimated_total_m=estimated_total_m,
        )

    def partial(
        self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> Tensor:
        """Returns each rank's unreduced float32 contribution (``Partial``)."""
        if topk_ids.dtype != DType.int32 or topk_weights.dtype != DType.float32:
            raise ValueError("Routing requires int32 ids and float32 weights")
        bundles = self._local_bundles()
        routed = (
            mxfp4_ops.routed
            if self.expert_format is RoutedExpertFormat.MXFP4
            else nvfp4_ops.routed
        )
        partial = F.call_on_mesh(
            routed,
            self.mesh,
            out_specs=DeviceMapping(self.mesh, (Partial(),)),
        )(
            x,
            topk_ids,
            topk_weights,
            PerShard(b[0] for b in bundles),
            PerShard(b[1] for b in bundles),
            PerShard(b[2] for b in bundles),
            PerShard(
                rank * self.per_rank for rank in range(self.mesh.num_devices)
            ),
            self.swiglu_limit,
        )
        return partial


class SharedExpert(Module[[Tensor], Tensor]):
    """A dense MXFP8 expert with its own float32 asymmetric clamp.

    Args:
        hidden_dim: Token width.
        moe_dim: Intermediate width.
        mesh: Mesh for S0 w1/w3 and S1 w2; defaults to the context.
        swiglu_limit: Asymmetric clamp limit from training.
        data_parallel: Whether the mesh is the replica axis, so the rank
            runs the whole expert on its own rows.
    """

    w1: MXFP8Linear
    w3: MXFP8Linear
    w2: MXFP8Linear

    def __init__(
        self,
        hidden_dim: int,
        moe_dim: int,
        *,
        mesh: DeviceMesh | None = None,
        swiglu_limit: float | None = 10.0,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        mesh = mesh or defaults()[1]
        self.swiglu_limit = swiglu_limit
        # Under data parallelism the drafter runs this expert once per rank
        # on that rank's rows, where an all-reduce over the rank's one shard
        # is the identity: a split expert would return 1/N of its sum. The
        # weights stay whole and nothing reduces.
        self.data_parallel = data_parallel
        w1 = MXFP8Linear(hidden_dim, moe_dim)
        w3 = MXFP8Linear(hidden_dim, moe_dim)
        w2 = MXFP8Linear(moe_dim, hidden_dim)
        if not data_parallel:
            w1 = w1.shard(0, mesh)
            w3 = w3.shard(0, mesh)
            w2 = w2.shard(1, mesh)
        self.w1 = w1
        self.w3 = w3
        self.w2 = w2

    def forward(self, x: Tensor) -> Tensor:
        """Returns the unscaled float32 shared output.

        The result is reduced across TP shards, whole on each rank under data
        parallelism.
        """
        if self.data_parallel:
            return self.partial(x)
        return F.allreduce_sum(self.partial(x))

    def partial(self, x: Tensor) -> Tensor:
        """Returns each rank's unreduced float32 shared output (``Partial``)."""
        activated = nvfp4_ops.swiglu(
            self.w1(x), self.w3(x), self.swiglu_limit
        ).cast(DType.bfloat16)
        partial = self.w2(activated)
        # The generic cast cannot carry Partial and would reduce in bf16.
        partial = F.call_on_mesh(
            F.cast, partial.mesh, out_specs=partial.mapping
        )(partial, DType.float32)
        return partial


class StackedSharedExpert(Module[[Tensor], Tensor]):
    """The trunk's dense MXFP8 expert over the adapter's stacked ``w13``.

    ``w1`` and ``w3`` read the same input, so the weight adapter stacks
    them into one ``w13`` leaf (``[w1; w3]`` along N, interleaved per TP
    rank) and one GEMM computes both; the gate/up halves split out of the
    merged output, and the SwiGLU is the same elementwise kernel either
    way. The DSpark/drafter tree keeps the unfolded :class:`SharedExpert`
    (its adapter leaves stay ``w1``/``w3`` by design), so the trunk alone
    switches to this class.

    Args:
        hidden_dim: Token width.
        moe_dim: Intermediate width.
        mesh: Mesh for S0 w13 and S1 w2; defaults to the context.
        swiglu_limit: Asymmetric clamp limit from training.
    """

    w13: MXFP8Linear
    w2: MXFP8Linear

    def __init__(
        self,
        hidden_dim: int,
        moe_dim: int,
        *,
        mesh: DeviceMesh | None = None,
        swiglu_limit: float | None = 10.0,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        mesh = mesh or defaults()[1]
        self.swiglu_limit = swiglu_limit
        # Under data parallelism the rank runs the whole expert on its own
        # rows: the weights stay whole and nothing reduces.
        self.data_parallel = data_parallel
        w13 = MXFP8Linear(hidden_dim, 2 * moe_dim)
        w2 = MXFP8Linear(moe_dim, hidden_dim)
        if not data_parallel:
            w13 = w13.shard(0, mesh)
            w2 = w2.shard(1, mesh)
        self.w13 = w13
        self.w2 = w2

    def forward(self, x: Tensor) -> Tensor:
        """Returns the unscaled shared output, reduced across TP shards."""
        if self.data_parallel:
            return self.partial(x).cast(DType.bfloat16)
        return F.allreduce_sum(self.partial(x))

    def partial(self, x: Tensor, x_rows: Mxfp8Rows | None = None) -> Tensor:
        """Returns each rank's unreduced float32 shared output (``Partial``).

        ``x_rows``, when the caller has ``x`` already quantized, feeds the
        ``w13`` GEMM in place of its own quantize.
        """
        gate_up = (
            self.w13(x)
            if x_rows is None
            else self.w13.forward_quantized(x_rows)
        )
        partial = self.w2.forward_swiglu(gate_up, self.swiglu_limit)
        # The generic cast cannot carry Partial and would reduce in bf16.
        partial = F.call_on_mesh(
            F.cast, partial.mesh, out_specs=partial.mapping
        )(partial, DType.float32)
        return partial
