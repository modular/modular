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
"""Pins that permissive arch configs store the resolved max sequence length.

``ArchConfigWithPermissiveMaxSeqLen.get_max_seq_len`` returns the stored
``max_position_embeddings`` field, so configs using the mixin must store the
resolved value (the user's ``max_length`` when set), not the raw checkpoint
bound — otherwise the tokenizer bound and KV-cache sizing ignore
``--max-length``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from max.driver import DeviceSpec
from max.pipelines.architectures.olmo2_modulev3.model_config import (
    Olmo2Config,
)
from max.pipelines.lib import MemoryEstimator
from test_common.mocks import DummyPipelineConfig
from transformers.models.olmo2.configuration_olmo2 import (
    Olmo2Config as HFOlmo2Config,
)


def _olmo2_arch_config(user_max_length: int | None) -> Olmo2Config:
    pipeline_config = DummyPipelineConfig(
        model_path="allenai/OLMo-2-1124-7B",
        quantization_encoding="bfloat16",
        max_batch_size=1,
        max_length=user_max_length,
        device_specs=[DeviceSpec.cpu()],
    )
    hf_config = HFOlmo2Config()
    if getattr(hf_config, "rope_parameters", None) is None:
        hf_config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": hf_config.rope_theta,
        }
    pipeline_config.model._huggingface_config = hf_config
    pipeline_config.model.weight_path = [Path("model.safetensors")]
    return Olmo2Config.initialize(pipeline_config)


@pytest.mark.parametrize("user_max_length", [None, 64])
def test_olmo2_modulev3_stores_resolved_max_seq_len(
    user_max_length: int | None,
) -> None:
    arch_config = _olmo2_arch_config(user_max_length)

    expected = (
        user_max_length
        if user_max_length is not None
        else HFOlmo2Config().max_position_embeddings
    )
    assert arch_config.get_max_seq_len() == expected


def test_olmo2_modulev3_kv_sizing_honors_max_length() -> None:
    """KV sizing derives its per-sequence bound from ``get_max_seq_len()``;
    with ``--max-length`` set below the checkpoint bound, the estimate must be
    sized for the user value, not the full bound."""
    ample_memory = 1 << 40

    bounded = MemoryEstimator._calculate_kv_cache_size(
        _olmo2_arch_config(64),
        max_batch_size=1,
        available_kv_cache_memory=ample_memory,
    )
    unbounded = MemoryEstimator._calculate_kv_cache_size(
        _olmo2_arch_config(None),
        max_batch_size=1,
        available_kv_cache_memory=ample_memory,
    )

    assert 0 < bounded < unbounded
