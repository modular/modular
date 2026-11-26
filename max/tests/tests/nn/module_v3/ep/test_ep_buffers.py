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
"""Tests the expert-parallel graph inputs and the buffers a decoder reads."""

from __future__ import annotations

import dataclasses

import pytest
from max import tree
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn import ep
from max.nn._graph_ops import ep as ep_kernels
from max.nn.ep.init import EPInitGraph, payload_buffers, payload_types
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.realization_context import GraphRealizationContext
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
)
from max.tensor.tensor import realization_context

_N = 2


def _config(
    eplb: bool = False,
    count: bool = False,
    n_devices: int = _N,
) -> ep.EPConfig:
    return ep.EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=256,
        top_k=2,
        n_experts=4,
        max_tokens_per_rank=4,
        n_gpus_per_node=n_devices,
        n_nodes=1,
        eplb_enabled=eplb,
        eplb_count_expert_load=count,
        num_moe_layers=2,
    )


def _read(
    graph: Graph, config: ep.EPConfig, mesh: DeviceMesh, first: int = 0
) -> ep.EPBuffers:
    """Reads the buffers from the graph inputs that start at ``first``."""
    leaves = [Tensor.from_graph_value(v) for v in graph.inputs[first:]]
    _, treedef = tree.flatten(payload_types(config))
    return payload_buffers(tree.unflatten(treedef, leaves), mesh)


@pytest.mark.parametrize("count", [False, True], ids=["no_count", "count"])
@pytest.mark.parametrize("eplb", [False, True], ids=["no_eplb", "eplb"])
def test_payload_types_are_the_serving_engines_inputs(
    eplb: bool, count: bool
) -> None:
    payload = payload_types(_config(eplb, count))

    assert list(payload) == [
        "atomic_counters",
        "send_buf_ptrs",
        "recv_buf_ptrs",
        "recv_count_ptrs",
        *(["eplb_log2phy", "eplb_logcnt"] if eplb else []),
        *(["eplb_expert_load"] if count else []),
    ]
    gpus = [DeviceRef.GPU(i) for i in range(_N)]
    assert [t.device for t in payload["atomic_counters"]] == gpus * 2
    for name in ("send_buf_ptrs", "recv_buf_ptrs", "recv_count_ptrs"):
        assert [t.device for t in payload[name]] == [DeviceRef.CPU()] * 2


@pytest.mark.parametrize("count", [False, True], ids=["no_count", "count"])
@pytest.mark.parametrize("eplb", [False, True], ids=["no_eplb", "eplb"])
def test_payload_buffers_read_each_input_in_place(
    mesh: DeviceMesh, eplb: bool, count: bool
) -> None:
    config = _config(eplb, count)
    graph = Graph("ep", input_types=tree.leaves(payload_types(config)))
    with GraphRealizationContext(graph) as ctx, realization_context(ctx):
        buffers = _read(graph, config, mesh)

    inputs = [value._mlir_value for value in graph.inputs]
    positions = [
        inputs.index(value._mlir_value)
        for leaf in tree.leaves(buffers, leaf=Tensor)
        for value in leaf.graph_values
    ]
    # Inputs 2 and 3 are the second group's counters, which only ep_init
    # writes; 4 to 9 are the send, receive and count tables of each group.
    tail = range(10, 10 + 2 * _N * eplb + _N * count)
    assert positions == [0, 1, 4, 6, 8, 5, 7, 9, *tail]
    assert buffers.counters.mesh == mesh
    assert not buffers.dispatch.send.is_distributed
    assert (buffers.eplb is not None) == eplb
    assert (buffers.expert_load is not None) == count


def test_payload_buffers_place_each_buffer_as_its_data(
    mesh: DeviceMesh,
) -> None:
    config = _config(eplb=True, count=True)
    graph = Graph("ep", input_types=tree.leaves(payload_types(config)))
    with GraphRealizationContext(graph) as ctx, realization_context(ctx):
        buffers = _read(graph, config, mesh)

    assert buffers.counters.placements == (Sharded(0),)
    # Every device holds the same EPLB tables.
    assert buffers.eplb is not None
    assert buffers.eplb.log2phy.placements == (Replicated(),)
    assert buffers.eplb.logcnt.placements == (Replicated(),)
    assert list(buffers.eplb.logcnt.shape) == [2, 4]
    # Each device tallies its own experts of every MoE layer.
    assert buffers.expert_load is not None
    assert buffers.expert_load.placements == (Sharded(1),)
    assert list(buffers.expert_load.shape) == [2, 4]


def test_buffers_move_to_a_models_mesh_of_the_same_devices(
    mesh: DeviceMesh,
) -> None:
    config = _config(eplb=True, count=True)
    graph = _exchange_graph(config)
    grid = DeviceMesh(mesh.devices, (1, _N), ("dp", "ep"))
    with GraphRealizationContext(graph) as ctx, realization_context(ctx):
        buffers = _read(graph, config, mesh, first=3 * _N)
        moved = buffers.on(grid)
        tokens, ids, _ = (
            Tensor.from_shard_values(
                [v.tensor for v in graph.inputs[i : i + _N]],
                DeviceMapping(grid, (Replicated(), Sharded(0))),
            )
            for i in range(0, 3 * _N, _N)
        )
        # The dispatch reads the counters on the model's mesh.
        ep.dispatch(_config(eplb=True), moved, tokens, ids)

    assert moved.counters.mesh == grid
    assert moved.counters.placements == (Sharded(0), Sharded(0))
    assert moved.eplb is not None
    assert moved.eplb.log2phy.placements == (Replicated(), Replicated())
    assert moved.expert_load is not None
    assert moved.expert_load.placements == (Sharded(1), Sharded(1))
    # The same graph values, re-placed; the host tables are unchanged.
    for before, after in zip(
        tree.leaves(buffers, leaf=Tensor),
        tree.leaves(moved, leaf=Tensor),
        strict=True,
    ):
        assert [v._mlir_value for v in before.graph_values] == [
            v._mlir_value for v in after.graph_values
        ]
    assert moved.dispatch is buffers.dispatch


def test_buffers_refuse_a_mesh_of_other_devices(mesh: DeviceMesh) -> None:
    config = _config()
    graph = Graph("ep", input_types=tree.leaves(payload_types(config)))
    swapped = DeviceMesh(mesh.devices[::-1], (_N,), ("ep",))
    with GraphRealizationContext(graph) as ctx, realization_context(ctx):
        buffers = _read(graph, config, mesh)
        with pytest.raises(ValueError, match="own devices"):
            buffers.on(swapped)


def _exchange_graph(config: ep.EPConfig) -> Graph:
    """Returns a graph of each device's tokens, ids and weights, then EP."""
    types = [
        TensorType(dtype, ["tokens", width], DeviceRef.GPU(i))
        for dtype, width in (
            (DType.bfloat16, config.hidden_size),
            (DType.int64, config.top_k),
            (DType.float32, config.top_k),
        )
        for i in range(_N)
    ]
    return Graph(
        "ep", input_types=[*types, *tree.leaves(payload_types(config))]
    )


def _replicated(graph: Graph, mesh: DeviceMesh) -> list[Tensor]:
    """Reads the tokens, ids and weights, the same on every device."""
    return [
        Tensor.from_shard_values(
            [v.tensor for v in graph.inputs[i : i + _N]],
            DeviceMapping(mesh, (Replicated(),)),
        )
        for i in range(0, 3 * _N, _N)
    ]


def test_allreduce_exchange_takes_any_integer_ids_and_sums_partially(
    mesh: DeviceMesh,
) -> None:
    config = dataclasses.replace(_config(), use_allreduce=True)
    graph = _exchange_graph(config)
    with GraphRealizationContext(graph) as ctx, realization_context(ctx):
        tokens, ids, weights = _replicated(graph, mesh)
        buffers = _read(graph, config, mesh, first=3 * _N)
        grouped, route = ep.dispatch_local(config, buffers, tokens, ids)
        out = ep.combine_local(
            config, buffers, grouped.tokens, route, weights, ids
        )

    assert out.placements == (Partial(),)


def test_exchange_refuses_what_it_cannot_serve(mesh: DeviceMesh) -> None:
    config = _config()
    counting = _config(count=True)
    graph = _exchange_graph(config)
    with GraphRealizationContext(graph) as ctx, realization_context(ctx):
        tokens, ids, weights = _replicated(graph, mesh)
        buffers = _read(graph, config, mesh, first=3 * _N)
        with pytest.raises(ValueError, match="MoE layer index"):
            ep.dispatch(counting, buffers, tokens, ids)
        with pytest.raises(ValueError, match=r"EPBuffers\.expert_load"):
            ep.dispatch(counting, buffers, tokens, ids, layer_idx=ids)
        by_device = DeviceMapping(mesh, (Sharded(0),))
        own_weights = Tensor.from_shard_values(
            list(weights.local_shards), by_device
        )
        with pytest.raises(ValueError, match="holds the same tokens"):
            ep.combine_local(
                dataclasses.replace(config, use_allreduce=True),
                buffers,
                tokens,
                ep.Route(src_info=ids),
                own_weights,
                ids,
            )
        with pytest.raises(ValueError, match="always exact"):
            ep.combine(
                dataclasses.replace(_config(), combine_mx_min_tokens=1),
                buffers,
                tokens,
                ep.Route(src_info=ids),
                tokens,
                weights,
            )


@pytest.mark.parametrize("fused_moe", [False, True])
def test_ep_init_graph_returns_each_devices_pointers(fused_moe: bool) -> None:
    init = EPInitGraph(dataclasses.replace(_config(), fused_moe=fused_moe))
    graph = init.trace(*init.input_types())

    assert graph.name == "ep_init"
    assert len(init.input_types()) == 2 * _N
    # The pointers and the ranks, then each device's fused MoE answer.
    assert len(graph.output_types) == _N + 1 + 2 * _N * fused_moe


def test_accepts_a_mesh_of_one_device_for_a_one_gpu_node(
    mesh: DeviceMesh,
) -> None:
    one = DeviceMesh(mesh.devices[:1], (1,), ("ep",))
    ep.check_supported(_config(n_devices=1), one)


def test_rejects_a_mesh_that_is_not_the_node(mesh: DeviceMesh) -> None:
    with pytest.raises(ValueError, match="n_gpus_per_node=4"):
        ep.check_supported(_config(n_devices=4), mesh)


_NVFP4 = QuantConfig.nvfp4(mlp_quantized_layers={0}, attn_quantized_layers={0})


@pytest.mark.parametrize("dyn_global_scales", [False, True])
def test_rejects_nvfp4_without_the_block_scaled_layout(
    monkeypatch: pytest.MonkeyPatch, mesh: DeviceMesh, dyn_global_scales: bool
) -> None:
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "hip")
    config = dataclasses.replace(
        _config(),
        dispatch_dtype=DType.uint8,
        dispatch_quant_config=_NVFP4,
        nvfp4_dyn_global_scales=dyn_global_scales,
    )
    with pytest.raises(ValueError, match="dispatches BF16, FP8, MXFP4, MXFP6"):
        ep.check_supported(config, mesh)
