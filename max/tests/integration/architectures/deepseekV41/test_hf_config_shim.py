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
"""Tests loading the published DeepSeek-V4.1-Flash config through the shim.

transformers does not know ``model_type: deepseek_v41``, so loading the
config needs the shim.
"""

from pathlib import Path

from max.pipelines.architectures import hf_config_shims  # noqa: F401
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"


def test_deepseekV41_config_loads() -> None:
    config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    assert config.model_type == "deepseek_v41"

    # Sub-configs are config objects, not dicts.
    text = config.text_config
    assert text.model_type == "deepseek_v41_text"
    assert text.hidden_size == 5120
    assert text.num_hidden_layers == 40
    assert text.n_routed_experts == 384
    assert text.vocab_size == 129280
    assert text.rms_norm_eps == 1e-20

    vision = config.vision_config
    assert vision.model_type == "deepseek_v41_vision"
    assert vision.hidden_size == 1024


def test_deepseekV41_text_config_yarn_standardization() -> None:
    """Checks the text sub-config loads with YaRN ``rope_scaling`` standardized.

    Standardization reads ``max_position_embeddings``, so the shim must bind
    it before ``PretrainedConfig.__init__`` runs.
    """
    config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    text = config.text_config
    assert text.max_position_embeddings == 1048576
    assert text.rope_scaling is not None
    assert text.rope_scaling["rope_type"] == "yarn"
    assert text.rope_scaling["factor"] == 16
