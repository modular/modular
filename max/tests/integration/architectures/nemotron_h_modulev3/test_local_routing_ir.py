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
"""Checks Nemotron-H's router and allreduce contract against the graph's IR.

Routing on every device is sound only while every reduction that feeds the
router input is bit-identical across ranks, and the kernels' global rank
order is a property of the plain ``mo.distributed.allreduce.sum`` op alone.
These checks read the dump a compile writes after MO PatternFusion and pin
that form for the production graph and for the trace the TP4 replica check
runs, which the block subgraph boundary has to leave alone. They also trace
the router's fused GEMV variant on four virtual devices, which a CPU mesh
cannot build.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_100a")
set_virtual_device_count(4)

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Module
from max.pipelines.architectures.nemotron_h_modulev3.layers.moe import (
    NemotronHRouter,
)
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    NemotronHConfig,
)
from max.pipelines.architectures.nemotron_h_modulev3.nemotron_h import (
    NemotronH,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from replica_probe import (
    LAYERS,
    NemotronHReplicaTrace,
    allreduce_ops,
    assert_allreduce_form,
    fused_allreduces,
    replica_config,
    replica_hf_config,
    stage_replica_trace,
)

NUM_DEVICES = 4
_BROADCAST_OP = re.compile(r"= mo\.distributed\.broadcast\(")


def _device_mesh() -> DeviceMesh:
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(NUM_DEVICES)),
        (NUM_DEVICES,),
        ("tp",),
    )


def _post_fusion_ir(
    graph: Graph, registry: dict[str, Tensor], ir_dir: Path
) -> str:
    """Compiles ``graph`` and returns its post-PatternFusion IR."""
    session = InferenceSession(devices=[CPU(), *_device_mesh().devices])
    ir_dir.mkdir(parents=True, exist_ok=True)
    session.debug.ir_output_dir = str(ir_dir)
    session.load(graph, weights_registry=registry)
    paths = sorted(ir_dir.glob("*.mo-pre-map.mlir"))
    assert paths, f"no post-PatternFusion IR dump in {ir_dir}"
    return "\n".join(path.read_text() for path in paths)


def _zeros(parameters: Iterable[tuple[str, Tensor]]) -> dict[str, Buffer]:
    """Returns one zero buffer per parameter, shared where shape and dtype
    match. These graphs read no weight value, only its type."""
    zeros: dict[tuple[tuple[int, ...], DType], Buffer] = {}
    weights: dict[str, Buffer] = {}
    for name, parameter in parameters:
        shape = tuple(int(dim) for dim in parameter.shape)
        key = (shape, parameter.dtype)
        if key not in zeros:
            raw = np.zeros(shape, f"u{parameter.dtype.size_in_bytes}")
            zeros[key] = Buffer.from_numpy(raw).view(
                parameter.dtype, list(shape)
            )
        weights[name] = zeros[key]
    return weights


def _production_graph(tmp_path: Path) -> tuple[Graph, dict[str, Tensor]]:
    """Stages the model the way the serving path does.

    The backbone wraps each whole block in ``as_subgraph``, which is the
    boundary under test.
    """
    refs = [DeviceRef.GPU(i) for i in range(NUM_DEVICES)]
    config = replica_config(refs)
    with F.lazy(), default_device(_device_mesh()), default_dtype(config.dtype):
        model = NemotronH(config)
    assert [kind.value for kind in model.backbone.layer_kinds] == LAYERS
    staged, registry = model.stage(
        *modulev3_ragged_kv_symbolic_inputs(
            kv_params=config.kv_params, device_refs=refs
        ),
        weights=_zeros(model.parameters),
    )
    return staged.graph, registry


def _trace_graph() -> tuple[Graph, dict[str, Tensor]]:
    """Stages the trace the TP4 replica check compiles."""
    refs = [DeviceRef.GPU(i) for i in range(NUM_DEVICES)]
    config = replica_config(refs)
    with F.lazy(), default_device(_device_mesh()), default_dtype(config.dtype):
        model = NemotronH(config)
    staged, registry = stage_replica_trace(NemotronHReplicaTrace(model), refs)
    return staged.graph, registry


def test_production_reduces_with_the_plain_op(tmp_path: Path) -> None:
    """The block subgraph keeps PatternFusion's chain from forming.

    A composite here would put the residual stream on a kernel whose add
    order is not the one the kernels' determinism test covers, so the router
    could no longer rely on it. One op is enough: the blocks of a kind share
    one subgraph definition, so its single reduction stands for all of them.
    """
    graph, registry = _production_graph(tmp_path)
    ir = _post_fusion_ir(graph, registry, tmp_path / "production")
    assert_allreduce_form(ir)
    ops = allreduce_ops(ir)
    assert ops, "the production graph never reduces"
    assert set(ops) == {"mo.distributed.allreduce.sum"}, ops
    assert not fused_allreduces(ir)


def test_replica_trace_reduces_as_production(tmp_path: Path) -> None:
    """The trace's reductions take production's form, one per layer.

    The trace has a subgraph per layer rather than one shared between them,
    so it carries one allreduce per layer instead of one definition, and each
    of them must be the plain op.
    """
    graph, registry = _trace_graph()
    ir = _post_fusion_ir(graph, registry, tmp_path / "trace")
    assert_allreduce_form(ir)
    ops = allreduce_ops(ir)
    assert len(ops) == len(LAYERS), ops
    assert set(ops) == {"mo.distributed.allreduce.sum"}, ops
    assert not fused_allreduces(ir)


class _RouterOnMesh(Module[..., tuple[Tensor, Tensor]]):
    """The router over a mesh, reading the input the way a block does."""

    def __init__(self, router: NemotronHRouter, mesh: DeviceMesh) -> None:
        self.router = router
        self.mesh = mesh

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Routes each device's replica of ``x``."""
        return self.router(x.to(self.mesh))


def _fused_router_graph() -> Graph:
    """Traces the router's fused variant on four devices.

    ``moe_sigmoid_gemv_router`` refuses anything but a CUDA or HIP device, so
    this cannot run on the CPU mesh ``test_model`` traces on. What it checks
    is the case the fused op was extended to cover: four devices each route
    their own replica of the router input, with no collective between them.
    The shape is one the kernel accepts: 128 routed experts, a whole number
    of warps, and top-2.
    """
    hf = replica_hf_config()
    hf.n_routed_experts = 128
    hf.num_experts_per_tok = 2
    refs = [DeviceRef.GPU(i) for i in range(NUM_DEVICES)]
    pipeline = Mock()
    pipeline.model.data_parallel_degree = 1
    pipeline.speculative = None
    config = NemotronHConfig.from_huggingface(
        hf,
        kv_params=NemotronHConfig.construct_kv_params(
            huggingface_config=hf,
            pipeline_config=pipeline,
            devices=refs,
            kv_cache_config=KVCacheConfig(),
            cache_dtype=DType.bfloat16,
        ),
        devices=refs,
        max_seq_len=1024,
    )
    config.fused_router = True
    with F.lazy(), default_device(_device_mesh()), default_dtype(config.dtype):
        router = _RouterOnMesh(NemotronHRouter(config), _device_mesh())
    staged, _ = router.stage(
        TensorType(config.dtype, ["rows", config.hidden_size], device=refs[0]),
        weights=_zeros(router.parameters),
    )
    return staged.graph


def test_fused_router_routes_on_every_device() -> None:
    """The fused gate GEMV and top-k cost no collective at four devices."""
    graph = _fused_router_graph()
    ir = str(graph)
    assert "mo.moe.sigmoid.gemv.single.group.router" in ir, ir
    broadcast = _BROADCAST_OP.findall(ir)
    assert not broadcast, broadcast
