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
"""Tests generation defaults for a checkpoint without a generation config.

With no ``generation_config.json``, the EOS token id must come from the top
level of ``config.json``, and the fallback sampling defaults must stay within
the recommended ``temperature=1.0`` and ``top_p`` of 0.95 or 1.0.
"""

from pathlib import Path
from unittest.mock import MagicMock

from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.context import (
    SamplingParams,
    SamplingParamsGenerationConfigDefaults,
    SamplingParamsInput,
)
from max.pipelines.lib.tokenizer import resolve_eos_token_ids
from transformers import AutoConfig, GenerationConfig

TESTDATA = Path(__file__).parent / "testdata"


def _real_config() -> AutoConfig:
    return AutoConfig.from_pretrained(str(TESTDATA / "config.json"))


def test_eos_token_id_resolves_from_real_checkpoint_config() -> None:
    """Checks ``eos_token_id`` resolves from the real config's top level."""
    pipeline_config = MagicMock()
    pipeline_config.model.huggingface_config = _real_config()
    pipeline_config.model.generation_config = GenerationConfig()
    pipeline_config.draft_model = None

    eos_token_ids = resolve_eos_token_ids(
        tokenizer_eos_token_id=None, pipeline_config=pipeline_config
    )
    assert eos_token_ids == {1}


def test_sampling_defaults_fall_back_within_recommended_envelope() -> None:
    """Checks fallback sampling defaults are within the recommended range."""
    params = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(),
        sampling_params_defaults=SamplingParamsGenerationConfigDefaults(),
    )
    assert params.temperature == 1.0
    assert params.top_p in (0.95, 1.0)
