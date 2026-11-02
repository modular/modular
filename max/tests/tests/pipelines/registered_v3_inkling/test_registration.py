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
from max.graph.weights import WeightData
from max.pipelines.architectures.inkling_modulev3 import weight_adapters
from max.pipelines.architectures.inkling_modulev3.model_config import (
    InklingConfig,
)
from max.pipelines.architectures.inkling_modulev3.registered_batch_processor import (
    _InklingInputs,
)
from max.pipelines.lib.role_contracts import VisionInputs


def test_expert_scale_weights_keep_role_and_quantization_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = NonCallableMock(spec=WeightData)
    original = {
        "layers.0.mlp.experts.w13_weight.weight_scale": value,
        "layers.0.mlp.experts.w2_weight.input_scale": value,
        "vision.encoder.weight": value,
    }
    monkeypatch.setattr(
        weight_adapters,
        "convert_safetensor_state_dict",
        lambda *args, **kwargs: original,
    )
    converted = weight_adapters.convert_registered_state_dict(
        {}, NonCallableMock()
    )
    assert set(converted) == {
        "decoder.layers.0.mlp.experts.w13_weight_scale",
        "decoder.layers.0.mlp.experts.w2_input_scale",
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
    assert set(recorded[0]) == set(original) - {"vision.encoder.weight"}
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
