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
"""The real checkpoint's keys, named by the weight rules, cover the modules.

``testdata/checkpoint_keys.txt`` lists the ``weight_map`` keys of
``model.safetensors.index.json`` at revision ``a4e59da52a7b``. A missing
parameter already fails ``Module.compile``; this also catches a checkpoint key
that lands under neither module, which the modules would silently skip.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Buffer, DeviceSpec
from max.dtype import DType
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.muse_glimmer import MuseGlimmerConfig
from max.pipelines.architectures.muse_glimmer.muse_glimmer import (
    MuseGlimmerTextModel,
)
from max.pipelines.architectures.muse_glimmer.vision import (
    MuseGlimmerVisionModel,
)
from max.pipelines.architectures.muse_glimmer.weight_adapters import (
    muse_glimmer_weight_rules,
)
from max.pipelines.lib import KVCacheConfig, WeightTransform
from max.tensor import default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"


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
def config() -> MuseGlimmerConfig:
    hf = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    config = MuseGlimmerConfig.initialize_from_config(
        _pipeline_config(), hf, max_seq_len=8192
    )
    assert config.text_config.num_hidden_layers == 52
    assert config.vision_config is not None
    assert config.vision_config.num_hidden_layers == 50
    return config


@pytest.fixture(scope="module")
def expected_names(config: MuseGlimmerConfig) -> set[str]:
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    with F.lazy(), default_dtype(DType.bfloat16):
        model = MuseGlimmerTextModel(config, mesh)
    return {f"language_model.{name}" for name, _ in model.parameters}


@pytest.fixture(scope="module")
def vision_names(config: MuseGlimmerConfig) -> set[str]:
    assert config.vision_config is not None
    with F.lazy(), default_dtype(DType.bfloat16):
        tower = MuseGlimmerVisionModel(
            config.vision_config,
            config.text_config.hidden_size,
            config.text_config.rms_norm_eps,
        )
    return set(dict(tower.parameters))


def _named(keys: list[str]) -> set[str]:
    placeholder = Buffer.from_numpy(np.zeros(1, dtype=np.float32))
    weights = SafetensorWeights(
        [],
        tensors=set(keys),
        tensors_to_file_idx={},
        _st_weight_map=dict.fromkeys(keys, placeholder),
    )
    hf = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    transform = WeightTransform(muse_glimmer_weight_rules(hf))
    return set(transform(dict(weights.items())))


def test_checkpoint_keys_cover_modules(
    checkpoint_keys: list[str],
    expected_names: set[str],
    vision_names: set[str],
) -> None:
    named = _named(checkpoint_keys)
    decoder = {
        n.removeprefix("decoder.") for n in named if n.startswith("decoder.")
    }
    vision = {
        n.removeprefix("vision.") for n in named if n.startswith("vision.")
    }
    assert decoder == expected_names
    assert vision == vision_names
    # Each key lands under exactly one of the two modules.
    assert len(decoder) + len(vision) == len(checkpoint_keys)
