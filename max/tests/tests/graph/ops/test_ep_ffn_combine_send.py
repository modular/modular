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
"""The EP-fused FFN send leaves no staging output in the MoE graph.

Builds one expert-parallel shard's local FFN twice, with the send fused and
unfused, and compares the emitted IR. The point of the fusion is a tensor that
is never allocated, so the only place to observe it is the graph: the kernel
gates prove the arithmetic, and the memory planner only predicts.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.nn.comm.ep import EPBatchManager, EPConfig, ep_kernels
from max.nn.moe import MoEQuantized
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

HIDDEN_DIM = 7168
MOE_DIM = 2048
NUM_EXPERTS = 64
TOP_K = 8
MAX_TOKENS_PER_RANK = 128
EP_SIZE = 2


def _nvfp4_quant_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
        # The send rides the fused-SwiGLU weight layout; without this the layer
        # predicate raises instead of answering.
        can_use_fused_swiglu=True,
    )


def _build_local_ep_compute(fuse: bool) -> str:
    """Returns the IR for one shard's local FFN, send fused or not."""
    quant_config = _nvfp4_quant_config()
    ep_config = EPConfig(
        dispatch_dtype=DType.uint8,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=MAX_TOKENS_PER_RANK,
        n_gpus_per_node=EP_SIZE,
        n_nodes=1,
        dispatch_quant_config=quant_config,
        fused_shared_expert=False,
        fuse_ffn_combine_send=fuse,
    )
    batch_manager = EPBatchManager(ep_config)
    devices = [DeviceRef.GPU(i) for i in range(EP_SIZE)]

    dispatch_types = ep_kernels._ep_dispatch_output_types(ep_config, devices[0])
    x_type = TensorType(DType.bfloat16, ["n_tokens", HIDDEN_DIM], devices[0])
    total_m_type = TensorType(DType.uint32, [], DeviceRef.CPU())

    moe = MoEQuantized(
        devices=[DeviceRef.CPU(), *devices],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=EP_SIZE,
        dtype=DType.uint8,
        apply_router_weight_first=False,
        ep_batch_manager=batch_manager,
        quant_config=quant_config,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(EP_SIZE)
    shard = moe.shard(devices)[0]
    # Gives every weight its qualified name; without it the per-expert scales
    # collide as a graph-wide `input_scale`. The zero values it fills in are
    # never read, since the graph is inspected rather than executed.
    moe.state_dict()

    fixed = [*dispatch_types, x_type, total_m_type]
    with Graph(
        "ep_local_ffn", input_types=[*fixed, *batch_manager.input_types()]
    ) as graph:
        dispatched = [v.tensor for v in graph.inputs[: len(dispatch_types)]]
        x = graph.inputs[len(dispatch_types)].tensor
        total_m = graph.inputs[len(dispatch_types) + 1].tensor
        batch_manager.fetch_buffers(graph.inputs[len(fixed) :])

        # What `ep_dispatch_wait` hands its caller: the kernel's outputs with
        # the trailing source-routing tensor stashed and replaced by the host
        # metadata. Spelled out because the dispatch itself is the one step of
        # the chain that needs a real accelerator.
        batch_manager._src_info[0] = dispatched[-1]
        expert_inputs = (
            *dispatched[:-1],
            batch_manager._common_grouped_matmul_metadata(),
        )

        down = shard._local_ep_compute(expert_inputs, x, total_m)
        if down is None:
            graph.output()
        else:
            graph.output(down)
    return str(graph)


@pytest.fixture(autouse=True)
def _target_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe the dispatch layout for the GPUs the graph is built against.

    ``accelerator_api`` reports the HOST's API, which is ``cpu`` here, so the
    dispatch helper would pick a layout no NVIDIA deployment uses.
    """
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "cuda")


def test_fused_send_elides_the_ffn_staging_output() -> None:
    """One flag, two graphs, and the down projection's output in only one.

    Everything else about the layer is held fixed, so the tensor's absence is
    attributable to the send rather than to any other difference.
    """
    unfused = _build_local_ep_compute(fuse=False)
    fused = _build_local_ep_compute(fuse=True)

    max_recv_tokens = MAX_TOKENS_PER_RANK * min(NUM_EXPERTS, EP_SIZE * TOP_K)
    staging = f"[{max_recv_tokens}, {HIDDEN_DIM}], bf16"
    elided = max_recv_tokens * HIDDEN_DIM * DType.bfloat16.size_in_bytes
    print(f"\nFFN staging output elided: {elided / 1024**2:.1f} MiB/device")

    # A type string recurs at every use, so presence is the honest test; the
    # op symbols are emitted once each and can be counted.
    assert staging in unfused, unfused
    assert "mo.composite.grouped_matmul_block_scaled" in unfused
    assert 'symbol = "mega_ffn.ep_combine_send"' not in unfused

    assert staging not in fused, fused
    assert "mo.composite.grouped_matmul_block_scaled" not in fused
    assert fused.count('symbol = "mega_ffn.ep_combine_send"') == 1, fused
