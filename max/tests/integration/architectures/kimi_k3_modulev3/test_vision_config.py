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
"""``KimiK3VisionConfig`` read through the registered HF config shim."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from max.driver import DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.hf_config_shims import KimiK3HFConfig
from max.pipelines.architectures.kimi_k3_modulev3.vision_config import (
    KimiK3VisionConfig,
    has_vision_tower,
)

_VISION_CONFIG: dict[str, Any] = {
    "patch_size": 14,
    "vt_hidden_size": 1024,
    "qkv_hidden_size": 1536,
    "vt_intermediate_size": 4096,
    "vt_num_attention_heads": 12,
    "vt_num_hidden_layers": 27,
    "init_pos_emb_height": 64,
    "init_pos_emb_width": 64,
    "init_pos_emb_time": 4,
    "merge_kernel_size": [2, 2],
    "mm_hidden_size": 1024,
    "text_hidden_size": 7168,
    "projector_ln_eps": 1e-5,
}


def _pipeline_config() -> Any:
    return SimpleNamespace(
        model=SimpleNamespace(
            device_specs=[DeviceSpec(id=0, device_type="cpu")]
        )
    )


@pytest.mark.parametrize(
    "extra", [{}, {"vision_config": None}], ids=["omitted", "null"]
)
def test_text_only_checkpoint_has_no_vision_config(
    extra: dict[str, Any],
) -> None:
    huggingface_config = KimiK3HFConfig(text_config={}, **extra)

    assert not has_vision_tower(huggingface_config)
    assert (
        KimiK3VisionConfig.initialize_from_config(
            _pipeline_config(), huggingface_config
        )
        is None
    )


def test_vision_checkpoint_reads_its_geometry() -> None:
    huggingface_config = KimiK3HFConfig(
        text_config={"hidden_size": 7168}, vision_config=_VISION_CONFIG
    )

    config = KimiK3VisionConfig.initialize_from_config(
        _pipeline_config(), huggingface_config
    )

    assert has_vision_tower(huggingface_config)
    assert config is not None
    assert config.dtype == DType.bfloat16
    assert config.devices == [DeviceRef.CPU()]
    assert config.patch_size == 14
    assert config.qkv_hidden_size // config.vt_num_attention_heads == 128
    assert config.merge_kernel_size == (2, 2)
    assert config.text_hidden_size == 7168
