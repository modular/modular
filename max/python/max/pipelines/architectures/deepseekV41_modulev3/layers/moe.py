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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMapping, DeviceMesh, Partial
from max.experimental.sharding.action import PerShard
from max.experimental.tensor import Tensor, default_device, defaults
from max.nn import Module
from max.nn.common_layers.quant_tensor import (
    NVFP4Tensor,
)
from max.nn.sequential import ModuleList
from max.nn.transparent_module import TransparentModule

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

    Drops its parent attribute name so leaves stay ``ffn.experts.E.wN``.

    Args:
        hidden_dim: Token width.
        moe_dim: Expert intermediate width.
        num_experts: Global expert count (384 for the full checkpoint).
        mesh: One-dimensional tensor-parallel mesh; defaults to the context.
        swiglu_limit: Asymmetric clamp limit from training.
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
    ) -> None:
        super().__init__()
        self.mesh = mesh or defaults()[1]
        self.hidden_dim = hidden_dim
        self.moe_dim = moe_dim
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
        the layout the grouped matmuls read.
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
