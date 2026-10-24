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
"""Tests registry wiring for DeepSeek-V4.1-Flash.

The target architecture resolves by name. The unified DSpark speculator is
not an architecture of its own: it is indexed against its target and reached
through ``select_speculator``, as the registry resolves a speculative pair.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.deepseekV41 import weight_adapters
from max.pipelines.architectures.deepseekV41.model import DeepseekV41Model
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
    dspark_draft_width,
)
from max.pipelines.architectures.unified_dspark_deepseekV41 import (
    UnifiedDSparkDeepseekV41BatchProcessor,
    UnifiedDSparkDeepseekV41Model,
)
from max.pipelines.lib.arch_lookup import select_speculator
from max.pipelines.lib.reasoning import get_parser_cls


def test_deepseekV41_arch_registered() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("DeepseekV41ForCausalLM")
    assert arch is not None
    assert arch.pipeline_model is DeepseekV41Model
    assert arch.config is DeepseekV41Config
    assert arch.multi_gpu_supported is True
    assert DeepseekV41Config.DEFAULT_ENCODING == arch.default_encoding
    assert DeepseekV41Config.SUPPORTED_ENCODINGS == arch.supported_encodings
    assert "float8_e4m3fn" in arch.supported_encodings
    assert arch.reasoning_parser == "deepseekv4"
    assert get_parser_cls(arch.reasoning_parser) is not None


def test_unified_dspark_deepseekV41_speculator_registered() -> None:
    speculator = select_speculator("DeepseekV41ForCausalLM", "dspark", None)
    assert speculator is not None
    assert speculator.name == "UnifiedDSparkDeepseekV41ForCausalLM"
    assert speculator.pipeline_model is UnifiedDSparkDeepseekV41Model
    assert speculator.batching is UnifiedDSparkDeepseekV41BatchProcessor
    assert (
        speculator.weight_adapters[WeightsFormat.safetensors]
        is weight_adapters.convert_dspark_safetensor_state_dict
    )
    assert speculator.supports_device_graph_capture is True

    derived = speculator.derive()
    assert derived.pipeline_model is UnifiedDSparkDeepseekV41Model
    assert derived.supports_device_graph_capture is True
    # Inherited from the base, so `--speculative-method dspark` resolves
    # num_speculative_tokens to the checkpoint's block size.
    assert derived.checkpoint_draft_width is dspark_draft_width
    # Mixed prefill and decode steps verify the decode rows' drafts.
    assert speculator.supports_spec_decode_mixed_batches is True
    assert derived.supports_spec_decode_mixed_batches is True
