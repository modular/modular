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

"""Config compatibility test for Gemma 4 E*B checkpoints.

The google/gemma-4-E*B checkpoints ship ``"num_global_key_value_heads": null``
in ``text_config`` (Hugging Face semantics: null falls back to
``num_key_value_heads``). Exercises the config entry points the pipeline
framework calls with a local E4B config.json (no network), which previously
crashed with ``TypeError: unsupported operand type(s) for %: 'NoneType' and
'int'`` in ``construct_kv_params``.
"""

from pathlib import Path
from unittest.mock import Mock

from max.driver import DeviceSpec
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
    _resolve_num_global_kv_heads,
)
from max.pipelines.lib import KVCacheConfig
from transformers import AutoConfig

CONFIG_DIR = Path(__file__).parent / "configs" / "gemma4_e4b"


def _load_hf_config() -> AutoConfig:
    return AutoConfig.from_pretrained(str(CONFIG_DIR), trust_remote_code=True)


def _mock_pipeline_config() -> Mock:
    """Builds a minimal PipelineConfig mock for config init."""
    model = Mock()
    model.kv_cache.cache_dtype = DType.bfloat16
    model.quantization_encoding = "bfloat16"
    model.weight_path = [Path("model.safetensors")]
    model.rope_type = "default"
    model.device_specs = [DeviceSpec.cpu()]
    model.max_length = None
    model.data_parallel_degree = 1

    pipeline_config = Mock()
    pipeline_config.model = model
    pipeline_config.speculative = None
    return pipeline_config


def test_resolve_num_global_kv_heads() -> None:
    """null/absent falls back to num_key_value_heads; explicit value wins."""
    null_config = Mock(num_global_key_value_heads=None, num_key_value_heads=2)
    assert _resolve_num_global_kv_heads(null_config) == 2

    explicit_config = Mock(
        num_global_key_value_heads=4, num_key_value_heads=2
    )
    assert _resolve_num_global_kv_heads(explicit_config) == 4


def test_construct_kv_params_with_null_global_kv_heads() -> None:
    """construct_kv_params must not crash on the E4B config -- the path that
    raised TypeError before the null fallback existed."""
    hf_config = _load_hf_config()
    pipeline_config = _mock_pipeline_config()

    kv_params = Gemma4ForConditionalGenerationConfig.construct_kv_params(
        hf_config,
        pipeline_config,
        [DeviceRef.CPU()],
        KVCacheConfig(),
        DType.bfloat16,
    )
    assert kv_params is not None


def test_text_config_init_with_null_global_kv_heads() -> None:
    """initialize_from_config resolves null to num_key_value_heads."""
    hf_config = _load_hf_config()
    pipeline_config = _mock_pipeline_config()

    text_config = Gemma4TextConfig.initialize_from_config(
        pipeline_config, hf_config.text_config
    )
    assert text_config.num_global_key_value_heads == 2
    assert (
        text_config.num_global_key_value_heads
        == text_config.num_key_value_heads
    )
