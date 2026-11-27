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
from max.pipelines.architectures.inkling_modulev3.model_config import (
    InklingConfig,
)
from max.pipelines.architectures.inkling_modulev3.preprocess_graph import (
    InklingImagePreprocess,
)
from max.pipelines.architectures.inkling_modulev3.registered_batch_processor import (
    _InklingInputs,
)
from max.pipelines.architectures.inkling_modulev3.weight_adapters import (
    inkling_modulev3_weight_rules,
)
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
        inkling_modulev3_weight_rules(NonCallableMock())
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


def test_rules_preserve_stored_projection_shapes_and_buffers() -> None:
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
    transformed = WeightTransform(
        inkling_modulev3_weight_rules(NonCallableMock())
    ).apply_data(original)
    assert len(transformed) == len(original)
    assert {id(value) for value in transformed.values()} == {
        id(value) for value in original.values()
    }


def test_preprocess_returns_named_raw_patch_rows() -> None:
    with Graph(
        "inkling_preprocess_contract",
        input_types=image_preprocess_input_types(),
    ) as graph:
        result = InklingImagePreprocess(40, DType.int16)(
            graph.inputs[0].tensor, graph.inputs[1].tensor
        )
        patches = result["patches"]
        assert isinstance(patches, VisionRows)
        pixels = patches["pixels"]
        assert isinstance(pixels, TensorValue)
        assert pixels.dtype == DType.int16
        assert tuple(pixels.shape[1:]) == (40, 40, 3)
        assert pixels.device == DeviceRef.CPU()
        graph.output(pixels)
