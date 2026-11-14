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
"""Checks MiMo's allreduce contract against the graph's IR.

Routing on every device is sound only while every reduction that feeds the
router input is bit-identical across ranks, so these checks pin the ops the
graph reduces with, the broadcasts the router does, and the gate that puts
routing on every device, at four ranks and at two. The forward and verify
checks read the dump a compile writes after MO PatternFusion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import tiny_fixture
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_100a")
set_virtual_device_count(4)

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import SafetensorWeights, WeightData
from max.nn import Module
from max.nn import _graph_ops as kernels
from max.nn.ep_config import EPConfig
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import (
    MiMoV2Decoder,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.quant import (
    MXFP8_DISPATCH_QUANT_CONFIG,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.module_registration import (
    _DecoderGraph,
    decoder_input_types,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import StagedGraph
from max.tensor.sharding import DeviceMesh
from mimo_v3_runner import kv_params
from replica_probe import (
    allreduce_ops,
    assert_allreduce_form,
    fused_allreduces,
)
from transformers.configuration_utils import PretrainedConfig

NUM_DEVICES = 4
_BROADCAST_OP = re.compile(r"= mo\.distributed\.broadcast\([^)]+")


def _device_mesh(num_devices: int = NUM_DEVICES) -> DeviceMesh:
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(num_devices)),
        (num_devices,),
        ("tp",),
    )


def _ep_manager(config: MiMoV2Config) -> EPBatchManager:
    config.ep_config = EPConfig(
        dispatch_dtype=DType.float8_e4m3fn,
        dispatch_quant_config=MXFP8_DISPATCH_QUANT_CONFIG,
        combine_dtype=DType.bfloat16,
        hidden_size=config.hidden_size,
        top_k=config.num_experts_per_tok,
        n_experts=config.n_routed_experts,
        max_tokens_per_rank=512,
        n_gpus_per_node=NUM_DEVICES,
        n_nodes=1,
        use_allreduce=True,
    )
    return EPBatchManager(config.ep_config)


def _post_fusion_ir(
    checkpoint: Path,
    ir_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    layout: str = "tp_tp",
    verify: bool = False,
) -> tuple[str, MiMoV2Config]:
    """Stages the requested tiny four-device graph and reads post-MO IR."""
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    hf = PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )
    refs = [DeviceRef.GPU(i) for i in range(NUM_DEVICES)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    config.return_logits = ReturnLogits.LAST_TOKEN
    if verify:
        config.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        config.target_layer_ids = [1, 3]
    ep_batch_manager = _ep_manager(config) if layout == "tp_ep" else None
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    adapted = convert_safetensor_state_dict(dict(weights.items()))
    mesh = _device_mesh()
    model: MiMoV2Decoder
    graph: Graph
    registry: dict[str, Tensor]
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        if verify:
            assert layout == "tp_tp"
            model = MiMoV2Decoder(config)
            verify_graph, registry = _stage_verify(model, adapted)
            graph = verify_graph.graph
        else:
            model = MiMoV2Decoder(config)
            inputs = decoder_input_types(
                config.kv_params,
                refs[0],
                ep=(
                    ep_batch_manager.input_tree()
                    if ep_batch_manager is not None
                    else None
                ),
                input_layout=MiMoV2Decoder.input_layout,
            )
            forward_graph, registry = _DecoderGraph(model).stage(
                inputs,
                weights=adapted,
                name=MiMoV2Decoder.graph_name,
            )
            graph = forward_graph.graph
    assert model.route_on_every_device

    session = InferenceSession(devices=[CPU(), *mesh.devices])
    ir_dir.mkdir(parents=True, exist_ok=True)
    session.debug.ir_output_dir = str(ir_dir)
    session.load(graph, weights_registry=registry)
    paths = sorted(ir_dir.glob("*.mo-pre-map.mlir"))
    assert paths, f"no post-PatternFusion IR dump in {ir_dir}"
    return "\n".join(path.read_text() for path in paths), config


class _VerifyGraph(Module[..., tuple[Tensor, list[Tensor]]]):
    def __init__(self, target: MiMoV2Decoder) -> None:
        self.target = target

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        sliding, full = self.target.kv_params.unflatten_basic_kv_tree(
            iter(x._graph_value for x in kv_inputs)
        )
        return self.target.verify(
            tokens,
            return_n_logits,
            input_row_offsets,
            {"sliding_attention": list(sliding), "full_attention": list(full)},
        )


def _stage_verify(
    target: MiMoV2Decoder,
    adapted: Mapping[str, WeightData],
) -> tuple[StagedGraph[..., tuple[Tensor, list[Tensor]]], dict[str, Tensor]]:
    # Verify's keyword/tree ABI differs from ModuleV3.forward; this wrapper
    # binds the target's verify path while retaining its model weights.
    staged, registry = _VerifyGraph(target).stage(
        *modulev3_ragged_kv_symbolic_inputs(
            kv_params=target.kv_params,
            device_refs=target.devices,
        ),
        weights={f"target.{name}": value for name, value in adapted.items()},
    )
    return staged, registry


def _assert_local_routing_ir(ir: str, layer_count: int) -> list[str]:
    """Asserts the reduction contract and returns the graph's broadcasts."""
    ops = allreduce_ops(ir)
    assert len(ops) == 2 * layer_count, (
        f"expected {2 * layer_count} allreduce-family ops, got "
        f"{len(ops)}: {ops}"
    )
    assert_allreduce_form(ir)
    assert fused_allreduces(ir), "the graph did not form an allreduce composite"
    return _BROADCAST_OP.findall(ir)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mimo_local_routing")
    tiny_fixture.write(path, tiny_fixture.tensors())
    return path


@pytest.mark.parametrize("layout", ["tp_tp", "tp_ep"])
def test_four_device_forward_ir_reduces_on_every_device(
    layout: str,
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_fusion, config = _post_fusion_ir(
        checkpoint, tmp_path / layout, monkeypatch, layout=layout
    )
    treatment_broadcasts = _assert_local_routing_ir(
        post_fusion, len(config.layer_types)
    )
    if layout == "tp_tp":
        # Every layer's attention boundary fuses: the norm in front of the MLP
        # or MoE is the composite's epilogue. Only the dense layer's boundary
        # takes the FP8 quantize that norm feeds along with it; a MoE layer's
        # quantize runs inside the shared MoE subgraph, behind its own
        # reductions, so those composites are the reduction, the residual add
        # and the norm alone.
        assert len(fused_allreduces(post_fusion)) == len(config.layer_types)
    # Two non-router broadcasts carry the graph inputs. The shared MoE
    # subgraph must not contain its two router-result broadcasts.
    assert len(treatment_broadcasts) == 2, treatment_broadcasts


def test_four_device_verify_ir_reduces_on_every_device(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    post_fusion, config = _post_fusion_ir(
        checkpoint, tmp_path, monkeypatch, verify=True
    )
    treatment_broadcasts = _assert_local_routing_ir(
        post_fusion, len(config.layer_types)
    )
    assert len(fused_allreduces(post_fusion)) == len(config.layer_types)
    # The verify graph carries the same two non-router input broadcasts.
    assert len(treatment_broadcasts) == 2, treatment_broadcasts


@pytest.mark.parametrize("num_devices", [4, 2], ids=["tp4", "tp2"])
def test_route_on_every_device_gate_covers_the_device_count(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_devices: int,
) -> None:
    """Both widths route locally: a two-way sum is order-independent too."""
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    hf = PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )
    refs = [DeviceRef.GPU(i) for i in range(num_devices)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    mesh = _device_mesh(num_devices)
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2Decoder(config)
    assert model.route_on_every_device
