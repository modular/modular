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
"""The real checkpoint's key list, renamed by the adapter, covers the module.

``testdata/checkpoint_keys.txt`` lists the ``weight_map`` keys of
``model.safetensors.index.json`` at revision ``a4e59da52a7b``. A missing
parameter already fails ``Module.compile``; this also catches two checkpoint
keys renamed onto one parameter, where the second silently wins.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import CPU, Buffer, DeviceSpec
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_dtype
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.muse_glimmer import MuseGlimmerConfig
from max.pipelines.architectures.muse_glimmer.muse_glimmer import (
    MuseGlimmerTextModel,
)
from max.pipelines.architectures.muse_glimmer.weight_adapters import (
    convert_safetensor_language_state_dict,
)
from max.pipelines.lib import KVCacheConfig
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"
VISION_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
)


def _pipeline_config() -> Mock:
    model = Mock()
    model.kv_cache = KVCacheConfig()
    model.quantization_encoding = "bfloat16"
    model.weight_path = [Path("model.safetensors")]
    model.max_length = None
    model.device_specs = [DeviceSpec.cpu()]
    model.data_parallel_degree = 1
    pipeline_config = Mock()
    pipeline_config.model = model
    pipeline_config.speculative = None
    return pipeline_config


@pytest.fixture(scope="module")
def checkpoint_keys() -> list[str]:
    return (TESTDATA / "checkpoint_keys.txt").read_text().split()


@pytest.fixture(scope="module")
def expected_names() -> set[str]:
    hf = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    config = MuseGlimmerConfig.initialize_from_config(
        _pipeline_config(), hf, max_seq_len=8192
    )
    assert config.text_config.num_hidden_layers == 52
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    with F.lazy(), default_dtype(DType.bfloat16):
        model = MuseGlimmerTextModel(config, mesh)
    return {f"language_model.{name}" for name, _ in model.parameters}


def _convert(keys: list[str]) -> set[str]:
    placeholder = Buffer.from_numpy(np.zeros(1, dtype=np.float32))
    weights = SafetensorWeights(
        [],
        tensors=set(keys),
        tensors_to_file_idx={},
        _st_weight_map=dict.fromkeys(keys, placeholder),
    )
    return set(convert_safetensor_language_state_dict(dict(weights.items())))


def test_checkpoint_keys_cover_module(
    checkpoint_keys: list[str], expected_names: set[str]
) -> None:
    converted = _convert(checkpoint_keys)
    assert converted == expected_names
    text_keys = [
        k for k in checkpoint_keys if not k.startswith(VISION_PREFIXES)
    ]
    assert len(text_keys) == len(converted)


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unexpected checkpoint key"):
        _convert(["model.mtp.weight"])
