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
"""DeepSeek-V4 ModuleV3: registration and the up-front config checks."""

from __future__ import annotations

from typing import Any
from unittest.mock import NonCallableMock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.outputs import ReturnLogits
from max.pipelines.architectures.deepseekV4_modulev3 import (
    deepseekV4_modulev3_arch,
)
from max.pipelines.architectures.deepseekV4_modulev3.model_config import (
    DeepseekV4Config,
)
from max.pipelines.architectures.deepseekV4_modulev3.reasoning import (
    DeepseekV4ReasoningParser,
)
from max.pipelines.architectures.deepseekV4_modulev3.tool_parser import (
    DeepseekV4ToolParser,
)
from max.pipelines.lib import reasoning, tool_parsing


def _config(**overrides: Any) -> DeepseekV4Config:
    fields: dict[str, Any] = dict(
        dtype=DType.bfloat16,
        kv_params=NonCallableMock(),
        devices=[DeviceRef.CPU(), DeviceRef.CPU()],
        max_seq_len=128,
        num_hidden_layers=2,
        compress_ratios=[4, 128],
        quantization_encoding="float8_e4m3fn",
    )
    fields.update(overrides)
    return DeepseekV4Config(**fields)


def test_arch_registers_its_parsers_and_fp8_only() -> None:
    arch = deepseekV4_modulev3_arch
    assert arch.name == "DeepseekV4ForCausalLM_ModuleV3"
    assert arch.supported_encodings == {"float8_e4m3fn"}
    name = "deepseekv4_modulev3"
    assert arch.reasoning_parser == arch.tool_parser == name
    assert reasoning.get_parser_cls(name) is DeepseekV4ReasoningParser
    assert tool_parsing.get_parser_cls(name) is DeepseekV4ToolParser


def test_supported_config_builds() -> None:
    _config()


@pytest.mark.parametrize(
    ("overrides", "names"),
    [
        ({"quantization_encoding": "float32"}, "quantization_encoding"),
        ({"expert_dtype": None}, "expert_dtype"),
        ({"expert_dtype": "fp8"}, "expert_dtype"),
        ({"data_parallel_degree": 2}, "data_parallel_degree"),
        ({"return_logits": ReturnLogits.ALL}, "enable_echo"),
    ],
)
def test_unsupported_setting_is_rejected(
    overrides: dict[str, Any], names: str
) -> None:
    with pytest.raises(ValueError, match=names):
        _config(**overrides)
