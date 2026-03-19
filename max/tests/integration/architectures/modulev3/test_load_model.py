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
"""load_model() integration tests for ModuleV3 architectures.

All models tested here use Llama-style configs and zero weights, so they
exercise module init (config parsing, weight adaptation, graph tracing)
without needing real checkpoints.
"""

from __future__ import annotations

import pytest
from max.pipelines.architectures.llama3_modulev3.model import Llama3Model
from max.pipelines.architectures.llama3_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.architectures.olmo_modulev3.model import OlmoModel
from max.pipelines.architectures.phi3_modulev3.model import Phi3Model
from test_common.load_model_helpers import (
    assert_load_model_succeeds,
    make_pipeline_config_factory,
    make_small_llama_config,
    make_zero_weights,
)


@pytest.mark.parametrize(
    "model_cls,repo_id",
    [
        (Llama3Model, "meta-llama/Llama-3.1-8B-Instruct"),
        (Phi3Model, "microsoft/phi-4"),
        (OlmoModel, "allenai/OLMo-1B-hf"),
        (Llama3Model, "ibm-granite/granite-3.1-8b-instruct"),
        (Llama3Model, "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"),
    ],
    ids=["llama3", "phi3", "olmo", "granite", "exaone"],
)
def test_load_model(model_cls: type, repo_id: str) -> None:
    hf_config = make_small_llama_config()
    weights = make_zero_weights(hf_config)
    make_pipeline_config = make_pipeline_config_factory(hf_config, repo_id)
    assert_load_model_succeeds(
        model_cls, make_pipeline_config, weights, convert_safetensor_state_dict
    )
