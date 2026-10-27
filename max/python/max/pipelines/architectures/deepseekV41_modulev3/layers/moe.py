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

"""V4.1 local routed experts and the tensor-parallel shared expert."""

from max import functional as F
from max.dtype import DType
from max.graph import TensorValue
from max.nn import Module
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_ops import (
    EPDispatchPayload,
    ep_requires_dispatch_scales,
    sigma_permute_gate_up_nvfp4,
)
from max.nn.common_layers.quant_tensor import (
    NVFP4Activation,
    NVFP4Tensor,
)
from max.nn.legacy.activation import GatedActivation
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.sequential import ModuleList
from max.nn.transparent_module import TransparentModule
from max.tensor import Tensor, default_device, defaults
from max.tensor.sharding import DeviceMapping, DeviceMesh, Partial
from max.tensor.sharding.action import PerShard

from . import nvfp4_ops
from .mxfp8_linear import MXFP8Linear, gate_up_op


class _NVFP4Linear(Module[[Tensor], Tensor]):
    weight: NVFP4Tensor

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = NVFP4Tensor.zeros((out_dim, in_dim))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Routed projections use grouped NVFP4 matmul")


class _Expert(Module[[], None]):
    w1: _NVFP4Linear
    w3: _NVFP4Linear
    w2: _NVFP4Linear

    def __init__(self, hidden_dim: int, moe_dim: int) -> None:
        super().__init__()
        self.w1 = _NVFP4Linear(hidden_dim, moe_dim)
        self.w3 = _NVFP4Linear(hidden_dim, moe_dim)
        self.w2 = _NVFP4Linear(moe_dim, hidden_dim)

    def forward(self) -> None:
        raise NotImplementedError("Experts are evaluated as local groups")


class RoutedExperts(TransparentModule[[Tensor, Tensor, Tensor], Tensor]):
    """Local NVFP4 W4A4 experts followed by float32 all-reduce.

    Under expert parallelism (:meth:`ep_forward`) the same local grouped
    chain runs on what each rank received through the P2P dispatch, and
    the combine returns every rank its own rows complete.

    Drops its parent attribute name so leaves stay ``ffn.experts.E.wN``.

    Args:
        hidden_dim: Token width.
        moe_dim: Expert intermediate width.
        num_experts: Global expert count (384 for the full checkpoint).
        mesh: One-dimensional tensor-parallel mesh; defaults to the context.
        swiglu_limit: Asymmetric clamp limit from training.
        ep_batch_manager: The EP batch manager when the routed experts run
            expert-parallel; ``None`` keeps the tensor-parallel path.
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
    ) -> None:
        super().__init__()
        self.mesh = mesh or defaults()[1]
        self.hidden_dim = hidden_dim
        self.moe_dim = moe_dim
        self.ep_batch_manager = ep_batch_manager
        if self.mesh.ndim != 1 or num_experts % self.mesh.num_devices:
            raise ValueError("Experts must divide evenly across a 1-D mesh")
        if hidden_dim % 32 or moe_dim % 32:
            raise ValueError("NVFP4 expert widths must be divisible by 32")
        self.num_experts = num_experts
        self.per_rank = num_experts // self.mesh.num_devices
        self.swiglu_limit = swiglu_limit
        experts = []
        for e in range(num_experts):
            device = self.mesh.devices[e // self.per_rank]
            with default_device(device):
                expert = _Expert(hidden_dim, moe_dim)
            # NVFP4Tensor.zeros puts the scalars on the host. The grouped
            # quantize kernel reads them on the device every step, so a host
            # copy would be freed under it; place them at load time instead.
            for linear in (expert.w1, expert.w3, expert.w2):
                tensor = linear.weight
                tensor.weight_scale_2 = tensor.weight_scale_2.to(device)
                tensor.input_scale = tensor.input_scale.to(device)
            experts.append(expert)
        self.experts = ModuleList(experts)
        # The dispatch quantizes every rank's activations with one uniform
        # scale: the max static ``w1``/``w3`` ``input_scale`` over all
        # experts, a pure function of the checkpoint the weight adapter
        # materializes as this leaf. Reading it here replaces the
        # in-graph cross-device max the EP forward used to run per step
        # -- two 4-byte scalar hubs whose gather/scatter lower to
        # device-wait pairs.
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
    ) -> list[tuple[NVFP4Tensor, NVFP4Tensor, NVFP4Tensor]]:
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
                    nvfp4_ops.stack([e.w1.weight for e in local]),
                    nvfp4_ops.stack([e.w3.weight for e in local]),
                    nvfp4_ops.stack([e.w2.weight for e in local]),
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
        through the P2P EP buffers, runs its local experts on what it
        received (the same NVFP4 W4A4 grouped chain as the TP path, with
        the training clamp), and combines the outputs back to the owning
        rank with the router weights applied. The result is each rank's
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
        if quant_config is None or not quant_config.is_nvfp4:
            raise ValueError(
                "the V4.1 EP path requires NVFP4 dispatch (the experts "
                "quantize their activations on device)"
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
        # The NVFP4 branch of ``per_device_tokens``, with each device's
        # activation carrying its own copy of the uniform scale (the shared
        # helper takes one scale for every device).
        assert (
            payload.scales is not None
            and payload.scales_offset is not None
            and global_scales is not None
        )
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

        activation = GatedActivation("silu", limit=self.swiglu_limit or 0.0)
        bundles = self._local_bundles()
        down_bundle = []
        for rank in range(self.mesh.num_devices):
            _w1, _w3, w2 = bundles[rank]
            # The fused gate/up kernel reads adjacent output columns as
            # (gate, up) pairs: stack each expert's ``[w1; w3]`` rows, then
            # sigma-permute (the ExpertParallelMoE weight layout).
            local = self.experts[
                rank * self.per_rank : (rank + 1) * self.per_rank
            ]
            concatenated: list[NVFP4Tensor] = []
            for expert in local:
                # concat_weights dispatches on the weight type, so the
                # all-NVFP4 inputs' result needs the runtime check.
                gate = quant_ops.concat_weights(
                    expert.w1.weight, expert.w3.weight, axis=0
                )
                assert isinstance(gate, NVFP4Tensor)
                concatenated.append(gate)
            gate_up = sigma_permute_gate_up_nvfp4(nvfp4_ops.stack(concatenated))
            down_in = quant_ops.grouped_matmul_silu(
                tokens[rank],
                gate_up,
                w2,
                payload.expert_start[rank],
                payload.expert_ids[rank],
                payload.usage_stats[rank]
                if payload.usage_stats is not None
                else None,
                quant_config,
                scales_offset=payload.scales_offset[rank]
                if payload.scales_offset is not None
                else None,
                estimated_total_m=estimated_total_m,
                activation=activation,
            )
            down_bundle.append(
                quant_ops.grouped_matmul(
                    down_in,
                    w2,
                    payload.expert_start[rank],
                    payload.expert_ids[rank],
                    payload.usage_stats[rank]
                    if payload.usage_stats is not None
                    else None,
                    scales_offset=payload.scales_offset[rank]
                    if payload.scales_offset is not None
                    else None,
                    estimated_total_m=estimated_total_m,
                )
            )
        combine_results = mgr.ep_combine_all(
            down_bundle, weight_shards, device_ids
        )
        out = Tensor.from_shard_values(
            [TensorValue(shard) for shard in combine_results], x.mapping
        )
        return out.cast(DType.bfloat16)

    def partial(
        self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> Tensor:
        """Returns each rank's unreduced float32 contribution (``Partial``)."""
        if topk_ids.dtype != DType.int32 or topk_weights.dtype != DType.float32:
            raise ValueError("Routing requires int32 ids and float32 weights")
        bundles = self._local_bundles()
        partial = F.call_on_mesh(
            nvfp4_ops.routed,
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
        amax_floor: Activation amax floor for all three dense projections.
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
        amax_floor: float = 1e-4,
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
        w1 = MXFP8Linear(hidden_dim, moe_dim, amax_floor=amax_floor)
        w3 = MXFP8Linear(hidden_dim, moe_dim, amax_floor=amax_floor)
        w2 = MXFP8Linear(moe_dim, hidden_dim, amax_floor=amax_floor)
        if not data_parallel:
            w1 = w1.shard(0, mesh)
            w3 = w3.shard(0, mesh)
            w2 = w2.shard(1, mesh)
        self.w1 = w1
        self.w3 = w3
        self.w2 = w2

    def forward(self, x: Tensor) -> Tensor:
        """Returns the unscaled float32 shared output, reduced across TP
        shards (whole on each rank under data parallelism)."""
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
        amax_floor: Activation amax floor for both dense projections.
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
        amax_floor: float = 1e-4,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        mesh = mesh or defaults()[1]
        self.swiglu_limit = swiglu_limit
        # Under data parallelism the rank runs the whole expert on its own
        # rows: the weights stay whole and nothing reduces.
        self.data_parallel = data_parallel
        w13 = MXFP8Linear(hidden_dim, 2 * moe_dim, amax_floor=amax_floor)
        w2 = MXFP8Linear(moe_dim, hidden_dim, amax_floor=amax_floor)
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

    def partial(self, x: Tensor) -> Tensor:
        """Returns each rank's unreduced float32 shared output (``Partial``)."""
        gate, up = gate_up_op(self.w13(x))
        activated = nvfp4_ops.swiglu(gate, up, self.swiglu_limit).cast(
            DType.bfloat16
        )
        partial = self.w2(activated)
        # The generic cast cannot carry Partial and would reduce in bf16.
        partial = F.call_on_mesh(
            F.cast, partial.mesh, out_specs=partial.mapping
        )(partial, DType.float32)
        return partial
