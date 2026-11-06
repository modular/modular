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
"""EPLB expert-load counting through the EP dispatch graph ops.

With ``eplb_count_expert_load`` the dispatch ops take their ``.expert_load``
form and add each local expert's routed-token count to a per-device
``[num_moe_layers, n_local_experts]`` tally, in the row of the dispatch's MoE
layer. The tally is never reset, so it holds running totals across executions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer, EPConfig

N_DEVICES = 2
NUM_MOE_LAYERS = 3


def _config() -> EPConfig:
    return EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=2048,
        top_k=8,
        n_experts=N_DEVICES * 32,
        max_tokens_per_rank=64,
        n_gpus_per_node=N_DEVICES,
        n_nodes=1,
        eplb_count_expert_load=True,
        num_moe_layers=NUM_MOE_LAYERS,
    )


def _build_graph(
    config: EPConfig, manager: EPBatchManager, split: bool
) -> Graph:
    token_types = [
        TensorType(
            DType.bfloat16, (f"len_{i}", config.hidden_size), DeviceRef.GPU(i)
        )
        for i in range(N_DEVICES)
    ]
    topk_types = [
        TensorType(DType.int32, (f"len_{i}", config.top_k), DeviceRef.GPU(i))
        for i in range(N_DEVICES)
    ]
    layer_types = [
        TensorType(DType.int32, [1], DeviceRef.GPU(i)) for i in range(N_DEVICES)
    ]
    with Graph(
        "ep_expert_load",
        input_types=[
            *token_types,
            *topk_types,
            *layer_types,
            *manager.input_types(),
        ],
    ) as graph:
        xs = [v.tensor for v in graph.inputs[:N_DEVICES]]
        topk = [v.tensor for v in graph.inputs[N_DEVICES : 2 * N_DEVICES]]
        layer_idx = [
            v.tensor for v in graph.inputs[2 * N_DEVICES : 3 * N_DEVICES]
        ]
        manager.fetch_buffers(graph.inputs[3 * N_DEVICES :])
        outputs: list[TensorValue] = []
        if split:
            for d in range(N_DEVICES):
                manager.ep_dispatch_async(xs[d], topk[d], d)
            for d in range(N_DEVICES):
                outputs.extend(manager.ep_dispatch_wait(d, layer_idx[d]))
        else:
            for result in manager.ep_dispatch_all(
                xs, topk, list(range(N_DEVICES)), layer_idx_per_device=layer_idx
            ):
                outputs.extend(TensorValue(v) for v in result)
        graph.output(*outputs)
    return graph


@pytest.mark.parametrize("split", [True, False], ids=["async_wait", "fused"])
def test_dispatch_adds_routed_tokens_to_the_layer_row(split: bool) -> None:
    if accelerator_count() < N_DEVICES:
        pytest.skip(f"needs {N_DEVICES} GPUs")
    devices = [Accelerator(i) for i in range(N_DEVICES)]
    session = InferenceSession(devices=[CPU(0), *devices])
    config = _config()
    n_local = config.n_experts // N_DEVICES

    initializer = EPCommInitializer(config)
    initializer.ep_init(session)
    manager = EPBatchManager(config)
    assert manager.input_tree()["eplb_expert_load"] is not None
    model = session.load(_build_graph(config, manager, split))

    expected = np.zeros((N_DEVICES, NUM_MOE_LAYERS, n_local), np.uint64)
    torch.manual_seed(7)
    # Layer 1 is never dispatched, so its row must stay zero.
    for layer in [0, 2, 0, 2]:
        lengths = torch.randint(1, config.max_tokens_per_rank, (N_DEVICES,))
        topk = [
            torch.topk(
                torch.randn(int(n), config.n_experts), config.top_k, dim=1
            ).indices.to(torch.int32)
            for n in lengths
        ]
        tokens = [
            Buffer.from_dlpack(
                torch.zeros(int(n), config.hidden_size, dtype=torch.bfloat16)
            ).to(devices[d])
            for d, n in enumerate(lengths)
        ]
        layer_inputs = [
            Buffer.from_numpy(np.array([layer], np.int32)).to(devices[d])
            for d in range(N_DEVICES)
        ]
        model.execute(
            *tokens,
            *[Buffer.from_dlpack(t).to(devices[d]) for d, t in enumerate(topk)],
            *layer_inputs,
            *initializer.model_inputs(),
        )

        routed = torch.cat(topk).flatten()
        counts = torch.bincount(routed, minlength=config.n_experts).numpy()
        for d in range(N_DEVICES):
            expected[d, layer] += counts[
                d * n_local : (d + 1) * n_local
            ].astype(np.uint64)
        for d in range(N_DEVICES):
            np.testing.assert_array_equal(
                initializer.expert_load[d].to(CPU()).to_numpy(),
                expected[d],
                err_msg=f"device {d} after a layer-{layer} dispatch",
            )
    assert not expected[:, 1].any()


def test_counting_needs_the_layer_index() -> None:
    manager = EPBatchManager(_config())
    with pytest.raises(ValueError, match="layer index"):
        manager._expert_load(0, None)


def test_counting_needs_the_layer_count() -> None:
    with pytest.raises(ValueError, match="num_moe_layers"):
        EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=2048,
            top_k=8,
            n_experts=64,
            max_tokens_per_rank=64,
            n_gpus_per_node=2,
            n_nodes=1,
            eplb_count_expert_load=True,
        )


def test_counting_off_declares_no_tally() -> None:
    config = _config()
    config.eplb_count_expert_load = False
    manager = EPBatchManager(config)
    assert "eplb_expert_load" not in manager.input_tree()
    assert manager._expert_load(0, None) is None
    tally_types = [
        t
        for t in manager.input_types()
        if isinstance(t, BufferType) and t.dtype == DType.uint64
    ]
    assert tally_types == []
