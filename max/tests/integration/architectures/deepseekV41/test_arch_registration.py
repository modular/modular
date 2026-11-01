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

The target architecture resolves by name. Under ``--speculative-method
dspark`` it pairs with the DSpark stages its checkpoint ships, as the
registry resolves a speculative pair.
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
from max.pipelines.lib.arch_lookup import ARCH_LOOKUP
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeVisionCachePipelineModel,
    spec_decode_architecture,
)
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


def test_dspark_pairs_with_the_stages_the_checkpoint_ships() -> None:
    draft = ARCH_LOOKUP.find("DSparkDeepseekV41DraftModel")
    assert draft is not None
    assert (
        draft.weight_adapters[WeightsFormat.safetensors]
        is weight_adapters.convert_dspark_safetensor_state_dict
    )

    paired = spec_decode_architecture("DeepseekV41ForCausalLM", "dspark", None)
    assert paired is not None
    assert paired.name == "DeepseekV41ForCausalLM+dspark"
    # Its images go through the pipeline's vision cache.
    assert paired.pipeline_model is _SpecDecodeVisionCachePipelineModel
    assert paired.supports_device_graph_capture is True
    # So `--speculative-method dspark` resolves num_speculative_tokens to the
    # checkpoint's block size.
    assert paired.checkpoint_draft_width is dspark_draft_width
    # Mixed prefill and decode steps verify the decode rows' drafts.
    assert paired.supports_spec_decode_mixed_batches is True
