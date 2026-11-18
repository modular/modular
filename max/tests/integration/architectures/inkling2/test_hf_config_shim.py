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
"""Checks the Inkling 2 config shim under both serializations of its fields.

The published checkpoint writes the reader's names (``sconv_kernel_size``);
a checkpoint saved through transformers' native config writes the names the
reference's ``attribute_map`` stores them under (``conv_kernel_size``).
"""

import pytest
from max.pipelines.architectures.hf_config_shims import _Inkling2HFConfig

READER_TEXT: dict[str, object] = {
    "sconv_kernel_size": 4,
    "intermediate_size": 2304,
    "model_max_length": 1048576,
    "sliding_window_size": 1024,
    "n_routed_experts": 384,
    "route_scale": 8.0,
}
READER_VISION: dict[str, object] = {
    "native_patch_size": 20,
    "adapter_merge_size": 2,
}


def _stored(
    reader: dict[str, object], aliases: dict[str, str]
) -> dict[str, object]:
    return {aliases[name]: value for name, value in reader.items()}


@pytest.mark.parametrize("serialization", ["reader", "stored"])
def test_reader_names_resolve(serialization: str) -> None:
    text: dict[str, object]
    vision: dict[str, object]
    if serialization == "reader":
        text, vision = READER_TEXT, READER_VISION
    else:
        text = _stored(READER_TEXT, _Inkling2HFConfig.text_aliases)
        vision = _stored(READER_VISION, _Inkling2HFConfig.vision_aliases)

    config = _Inkling2HFConfig(text_config=text, vision_config=vision)

    for name, value in READER_TEXT.items():
        assert getattr(config.text_config, name) == value, name
    for name, value in READER_VISION.items():
        assert getattr(config.vision_config, name) == value, name


def test_reader_name_wins_over_stored_name() -> None:
    config = _Inkling2HFConfig(
        text_config={"sconv_kernel_size": 4, "conv_kernel_size": 3}
    )
    assert config.text_config.sconv_kernel_size == 4
