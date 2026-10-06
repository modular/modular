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

"""Loads the published Muse Glimmer configs through the local shims."""

import json
from pathlib import Path

import pytest
from max.pipelines.architectures import hf_config_shims
from transformers import AutoConfig, PretrainedConfig

TESTDATA = Path(__file__).parent / "testdata"

TEXT_FULL = list(range(3, 52, 4))
VISION_FULL = [*range(3, 48, 4), 49]


def _stripped_config() -> PretrainedConfig:
    # The published config states both schedules, so drop them to make the
    # shim derive them.
    raw = json.loads((TESTDATA / "config.json").read_text())
    del raw["text_config"]["layer_types"]
    del raw["text_config"]["layer_rope_theta"]
    del raw["vision_config"]["layer_types"]
    return hf_config_shims.MuseGlimmerHFConfig(**raw)


@pytest.mark.parametrize(
    "config",
    [
        AutoConfig.from_pretrained(str(TESTDATA / "config.json")),
        _stripped_config(),
    ],
    ids=["published", "derived"],
)
def test_muse_glimmer_config(config: PretrainedConfig) -> None:
    assert config.model_type == "muse_glimmer"

    text = config.text_config
    full = [i for i, t in enumerate(text.layer_types) if t == "full_attention"]
    assert len(text.layer_types) == 52
    assert full == TEXT_FULL
    assert set(text.layer_types) == {"full_attention", "sliding_attention"}
    assert text.layer_rope_theta[3] == 0
    assert text.layer_rope_theta[0] == 500000.0

    vision = config.vision_config
    full = [
        i for i, t in enumerate(vision.layer_types) if t == "full_attention"
    ]
    assert len(vision.layer_types) == 50
    assert full == VISION_FULL
