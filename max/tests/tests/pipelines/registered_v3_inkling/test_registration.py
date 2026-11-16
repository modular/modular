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
"""Checks Inkling's role names and late vision-cache input attachment."""

from collections import OrderedDict
from typing import Any
from unittest.mock import NonCallableMock

import numpy as np
import pytest
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorValue
from max.graph.weights import WeightData
from max.nn.legacy.graph_groups import VisionRows
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.architectures.inkling.layers.moe import InklingMLP
from max.pipelines.architectures.inkling.model import InklingModel
from max.pipelines.architectures.inkling.preprocess_graph import (
    InklingImagePreprocess,
)
from max.pipelines.architectures.inkling.weight_adapters import (
    inkling_weight_rules,
)
from max.pipelines.architectures.inkling_modulev3 import weight_adapters
from max.pipelines.architectures.inkling_modulev3.model_config import (
    InklingConfig,
)
from max.pipelines.architectures.inkling_modulev3.registered_batch_processor import (
    _InklingInputs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.role_contracts import VisionInputs
from max.pipelines.lib.weight_transform import WeightTransform


def test_expert_scale_weights_keep_role_and_quantization_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = NonCallableMock(spec=WeightData)
    original = {
        "model.llm.layers.0.mlp.experts.w13_weight.scale": value,
        "model.llm.layers.0.mlp.experts.w2_weight.input_amax": value,
        "model.visual.encoder.weight": value,
    }
    converted = WeightTransform(
        weight_adapters.inkling_modulev3_weight_rules(NonCallableMock())
    )(original)
    assert set(converted) == {
        "decoder.layers.0.mlp.experts.w13_weight_scale",
        "decoder.layers.0.mlp.experts.w2_input_amax",
        "vision.encoder.weight",
    }
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        InklingConfig.__bases__[0],
        "finalize",
        lambda self, hf, state: recorded.append(state),
    )
    config = InklingConfig.__new__(InklingConfig)
    config.finalize(NonCallableMock(), converted)
    assert set(recorded[0]) == {
        "layers.0.mlp.experts.w13_weight.weight_scale",
        "layers.0.mlp.experts.w2_weight.input_amax",
    }
    assert all(weight is value for weight in recorded[0].values())


def test_late_cache_attachments_pack_only_device_zero_vision() -> None:
    buffers = [Buffer.from_numpy(np.array([i], np.int64)) for i in range(7)]
    _, structure = tree.flatten(
        OrderedDict(
            tokens="tokens",
            vision=VisionInputs(
                embeddings="embeddings", scatter_indices="indices"
            ),
        )
    )
    inputs = _InklingInputs(
        tokens=buffers[0],
        input_row_offsets=buffers[1],
        return_n_logits=buffers[2],
        input_tree_def=structure,
    )
    # The pipeline fills these after the batch processor has returned.
    inputs.vision_embeddings = buffers[3:5]
    inputs.vision_scatter_indices = buffers[5:7]
    assert inputs.buffers == (buffers[0], buffers[3], buffers[5])


@pytest.mark.parametrize("module_v3", [False, True])
def test_rules_preserve_stored_projection_shapes_and_buffers(
    module_v3: bool,
) -> None:
    weights = {
        "model.llm.layers.0.mlp.w13_dn.weight": np.zeros((16, 8), np.float32),
        "model.llm.layers.1.mlp.shared_experts.shared_w13_weight": np.zeros(
            (2, 8, 8), np.float32
        ),
        "model.llm.layers.1.mlp.shared_experts.shared_w2_weight": np.zeros(
            (2, 8, 4), np.float32
        ),
        "model.llm.layers.1.mlp.gate.weight": np.zeros((10, 8), np.float32),
        "model.llm.layers.1.mlp.experts.w13_weight.input_amax": np.zeros(
            (1,), np.float32
        ),
    }
    original = {
        name: WeightData.from_numpy(value, name)
        for name, value in weights.items()
    }
    rules = (
        weight_adapters.inkling_modulev3_weight_rules
        if module_v3
        else inkling_weight_rules
    )
    transformed = WeightTransform(rules(NonCallableMock())).apply_data(original)
    assert len(transformed) == len(original)
    assert {id(value) for value in transformed.values()} == {
        id(value) for value in original.values()
    }


def test_dense_mlp_declares_the_stored_fused_projections() -> None:
    module = InklingMLP(
        dtype=DType.bfloat16,
        hidden_dim=8,
        feed_forward_length=4,
        devices=[DeviceRef.CPU()],
    )
    weights = module.raw_state_dict()
    assert set(weights) == {"w13_dn.weight", "w2_md.weight"}
    assert tuple(weights["w13_dn.weight"].shape) == (8, 8)
    assert tuple(weights["w2_md.weight"].shape) == (8, 4)


def test_fused_pipeline_weight_entry_names_both_towers() -> None:
    context = NonCallableMock(spec=ModelLoadContext)
    context.adapter = WeightTransform(inkling_weight_rules(NonCallableMock()))
    original = {
        name: WeightData.from_numpy(np.zeros((1,), np.float32), name)
        for name in (
            "model.llm.embed.weight",
            "model.visual.layers.linear_0.weight",
        )
    }
    (entry,) = InklingModel.weight_entries(context, None)
    assert set(entry.transform.apply_data(original)) == {
        "embed.weight",
        "linears.0.weight",
    }


def test_preprocess_returns_named_raw_patch_rows() -> None:
    with Graph(
        "inkling_preprocess_contract",
        input_types=image_preprocess_input_types(),
    ) as graph:
        result = InklingImagePreprocess(40)(
            graph.inputs[0].tensor, graph.inputs[1].tensor
        )
        patches = result["patches"]
        assert isinstance(patches, VisionRows)
        pixels = patches["pixels"]
        assert isinstance(pixels, TensorValue)
        assert pixels.dtype == DType.int32
        assert tuple(pixels.shape[1:]) == (40, 40, 3)
        assert pixels.device == DeviceRef.CPU()
        graph.output(pixels)
