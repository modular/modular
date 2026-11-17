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
"""The real checkpoint's keys, named by the weight rules, cover the module.

``testdata/checkpoint_keys.txt`` lists the ``weight_map`` keys of
``model.safetensors.index.json`` for RecursiveSI/flint. A missing
parameter already fails ``Module.compile``; this also catches a checkpoint
key the module would silently skip.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from max import functional as F
from max.driver import Buffer, DeviceSpec
from max.dtype import DType
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.flint import FlintConfig
from max.pipelines.architectures.flint.flint import Flint
from max.pipelines.architectures.flint.weight_adapters import (
    flint_weight_rules,
)
from max.pipelines.lib import KVCacheConfig, WeightTransform
from max.tensor import default_dtype
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"


def _pipeline_config() -> Mock:
    """A pipeline config carrying only the fields ``FlintConfig`` reads."""
    model = Mock()
    model.huggingface_config = AutoConfig.from_pretrained(
        str(TESTDATA / "config.json")
    )
    model.kv_cache = KVCacheConfig()
    model.quantization_encoding = "bfloat16"
    # Encoding resolution walks the weight files when a path is set and the
    # repo's encodings otherwise; both are empty for a config-only fixture.
    model.weight_path = None
    model.huggingface_weight_repo = Mock(supported_encodings=[])
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
def config() -> FlintConfig:
    return FlintConfig.initialize(_pipeline_config(), max_seq_len=8192)


@pytest.fixture(scope="module")
def expected_names(config: FlintConfig) -> set[str]:
    with F.lazy(), default_dtype(DType.bfloat16):
        model = Flint(config)
    return {name for name, _ in model.parameters}


def _convert(keys: list[str]) -> set[str]:
    placeholder = Buffer.from_numpy(np.zeros(1, dtype=np.float32))
    weights = SafetensorWeights(
        [],
        tensors=set(keys),
        tensors_to_file_idx={},
        _st_weight_map=dict.fromkeys(keys, placeholder),
    )
    transform = WeightTransform(
        flint_weight_rules(_pipeline_config().model.huggingface_config)
    )
    return set(transform(dict(weights.items())))


def test_checkpoint_keys_cover_module(
    checkpoint_keys: list[str],
    expected_names: set[str],
) -> None:
    assert _convert(checkpoint_keys) == expected_names
