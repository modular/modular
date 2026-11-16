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
"""The ModuleV3 vision tower registers, loads the adapter's names and traces.

No checkpoint is loaded and nothing compiles: the tower is built at a tiny
width on the CPU, its weights are the weight adapter's output for a checkpoint
listing that names every tower weight, and its graph is staged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.glm5_next.arch import (
    glm5_next_arch,
)
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig,
)
from max.pipelines.architectures.glm5_next.modules import (
    Glm5NextDecoder,
)
from max.pipelines.architectures.glm5_next.vision.config import (
    Glm5NextVisionConfig,
)
from max.pipelines.architectures.glm5_next.vision.tower import (
    Glm5NextVisionEncoder,
)
from max.pipelines.architectures.glm5_next.weight_adapters import (
    VISION_WEIGHT_PREFIX,
    glm5_next_weight_rules,
)
from max.pipelines.lib import GraphRole, WeightTransform
from max.pipelines.lib.module_registration import VisionBatchProcessor
from max.pipelines.modeling.types import InputModality
from max.tensor import default_device
from transformers import AutoConfig

_HIDDEN_SIZE = 64

_VISION_CONFIG = Glm5NextVisionConfig(
    depth=2,
    hidden_size=32,
    num_heads=2,
    intermediate_size=48,
    projection_intermediate_size=40,
    out_hidden_size=_HIDDEN_SIZE,
    patch_size=2,
    temporal_patch_size=2,
    spatial_merge_size=2,
)


class _Weight:
    def __init__(self, data: WeightData) -> None:
        self._data = data

    def data(self) -> WeightData:
        return self._data


def _encoder() -> Glm5NextVisionEncoder:
    config = SimpleNamespace(
        vision_config=_VISION_CONFIG,
        devices=[DeviceRef.CPU()],
        hidden_size=_HIDDEN_SIZE,
    )
    with F.lazy(), default_device(CPU()):
        return Glm5NextVisionEncoder(cast(Glm5NextConfig, config))


def _checkpoint_name(name: str) -> str:
    """The checkpoint name a tower weight is read from."""
    return "model.visual." + name.replace("attn.qkv_proj.", "attn.qkv.")


def _checkpoint(encoder: Glm5NextVisionEncoder) -> dict[str, Weights]:
    """A bfloat16 checkpoint in the reference layout of every tower weight."""
    checkpoint: dict[str, Weights] = {}
    for name, weight in encoder.parameters:
        shape = [int(d) for d in weight.shape]
        data = WeightData.from_numpy(
            np.zeros(shape, np.float32), _checkpoint_name(name)
        ).astype(DType.bfloat16)
        checkpoint[_checkpoint_name(name)] = cast(Weights, _Weight(data))
    return checkpoint


def _tower_weights(encoder: Glm5NextVisionEncoder) -> dict[str, WeightData]:
    """The rules' output for the checkpoint, as the tower's role takes it."""
    hf_config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=1)
    )
    adapted = WeightTransform(
        glm5_next_weight_rules(cast(AutoConfig, hf_config))
    )(_checkpoint(encoder))
    return {
        name.removeprefix(VISION_WEIGHT_PREFIX): weight
        for name, weight in adapted.items()
    }


def test_the_architecture_registers_the_tower_for_images_and_video() -> None:
    arch = glm5_next_arch
    assert arch.modules[GraphRole.VISION] is Glm5NextVisionEncoder
    assert Glm5NextDecoder.consumes_vision
    assert arch.batching is not None
    assert issubclass(arch.batching, VisionBatchProcessor)
    assert {InputModality.IMAGE, InputModality.VIDEO} <= arch.input_modalities
    assert arch.config is Glm5NextConfig
    assert Glm5NextConfig.weight_prefixes[GraphRole.VISION] == (
        VISION_WEIGHT_PREFIX
    )


def test_the_rules_name_every_tower_weight_in_its_shape() -> None:
    encoder = _encoder()
    weights = _tower_weights(encoder)
    expected = {
        name: [int(d) for d in weight.shape]
        for name, weight in encoder.parameters
    }
    assert {
        name: [int(d) for d in weight.shape] for name, weight in weights.items()
    } == expected


def test_the_tower_traces_to_one_row_per_merged_patch() -> None:
    encoder = _encoder()
    weights = _tower_weights(encoder)
    with F.lazy(), default_device(CPU()):
        staged, _ = encoder.stage(*encoder.input_types(), weights=weights)
    (output,) = staged.graph.output_types
    assert isinstance(output, TensorType)
    assert output.dtype == DType.bfloat16
    assert int(output.shape[1]) == _HIDDEN_SIZE
    assert [value.type for value in staged.graph.inputs] == list(
        encoder.input_types()
    )
