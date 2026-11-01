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
"""The vision graph builds, with and without a tower replica per device.

Both plain serving (``graph_specs``) and the MTP pairing
(``build_vision_graph``) build it through ``Glm5NextModel._vision_spec``. A
classmethod that read ``self`` there once broke both, and no test traced it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next.model import Glm5NextModel
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig,
    Glm5NextVisionConfig,
)
from max.pipelines.architectures.glm5_next.nn.visual_transformer import (
    Glm5NextVisionTransformer,
)

HIDDEN_SIZE = 256
NUM_HEADS = 8

VISION_CONFIG = SimpleNamespace(
    depth=1,
    hidden_size=128,
    num_heads=2,
    intermediate_size=256,
    projection_intermediate_size=256,
    out_hidden_size=HIDDEN_SIZE,
    patch_size=14,
    temporal_patch_size=2,
    spatial_merge_size=2,
)


def _config(devices: list[DeviceRef]) -> Glm5NextConfig:
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=HIDDEN_SIZE,
        num_layers=1,
        page_size=128,
        devices=devices,
        num_q_heads=NUM_HEADS,
    )
    return Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=devices,
        max_seq_len=512,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=512,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        vision_config=Glm5NextVisionConfig.initialize_from_config(
            VISION_CONFIG, dtype=DType.bfloat16
        ),
    )


def _tower_weights(config: Glm5NextConfig) -> dict[str, WeightData]:
    """Zero weights under the post-adapter tower names."""
    assert config.vision_config is not None
    tower = Glm5NextVisionTransformer(config.vision_config, DeviceRef.CPU())
    return {
        name: WeightData.from_numpy(
            np.zeros([int(d) for d in w.shape.static_dims], np.float32), name
        ).astype(w.dtype)
        for name, w in tower.raw_state_dict().items()
    }


@pytest.mark.parametrize("replicate", [True, False])
def test_vision_graph_traces(
    monkeypatch: pytest.MonkeyPatch, replicate: bool
) -> None:
    monkeypatch.setattr(Glm5NextModel, "replicate_vision_tower", replicate)
    config = _config([DeviceRef.CPU(0), DeviceRef.CPU(1)])
    spec = Glm5NextModel._vision_spec(config, _tower_weights(config))
    graph = spec.trace()
    assert len(graph.output_types) == 2
