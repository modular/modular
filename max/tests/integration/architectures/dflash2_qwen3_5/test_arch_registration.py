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


"""How a Qwen3.5 target and a DFlash2 draft register and pair.

The draft registers as an architecture of its own with no pipeline model, and
the pairing resolves to the unified spec-decode model. The capability flags
the pairing declares must match what the graph can actually do: a wrongly
optimistic ``supports_device_graph_capture`` is a hang or a wrong answer at
serve time, not a config error.
"""

from __future__ import annotations

from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.dflash2_qwen3_5.model import (
    DFlash2Qwen3_5ArchConfig,
)
from max.pipelines.architectures.dflash2_qwen3_5.model_config import (
    dflash2_draft_width,
)
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.tokenizer import Qwen3_5Tokenizer
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    spec_decode_architecture,
)

TARGET = "Qwen3_5ForConditionalGeneration"
DRAFT = "DFlash2DraftModel"


def _fused() -> SupportedArchitecture:
    arch = spec_decode_architecture(TARGET, "dflash2", DRAFT)
    assert arch is not None
    return arch


def test_the_pairing_resolves_to_the_unified_model() -> None:
    arch = _fused()
    assert arch.name == f"{TARGET}+dflash2"
    assert arch.pipeline_model is not None
    assert issubclass(arch.pipeline_model, SpecDecodePipelineModel)
    assert arch.tokenizer is Qwen3_5Tokenizer
    assert arch.tool_parser == "qwen3_5"
    assert arch.reasoning_parser == "qwen3_5"
    # The target may be NVFP4; the drafter loads bf16 from its own checkpoint.
    assert {"bfloat16", "float4_e2m1fnx2"} <= arch.supported_encodings
    assert Qwen3_5Config.DEFAULT_ENCODING == arch.default_encoding
    assert Qwen3_5Config.SUPPORTED_ENCODINGS == arch.supported_encodings


def test_the_graph_keeps_the_name_the_engine_finds_it_by() -> None:
    assert _fused().spec_decode_graph_names["dflash2"] == (
        "qwen3_5_with_dflash2_graph"
    )


def test_the_checkpoint_fixed_draft_width_is_registered() -> None:
    """``PipelineConfig.from_args`` resolves an omitted
    ``--num-speculative-tokens`` only through this callback, so an unset one
    leaves the config's width disagreeing with the graph's."""
    assert _fused().checkpoint_draft_width is dflash2_draft_width


def test_the_declared_capabilities_match_what_the_graph_supports() -> None:
    """Capture is refused and TP is not wired.

    Capture is off until validated, as for the Qwen3.5 MTP graph. The
    drafter's body is unsharded and borrows the target's collective embedding
    and head, so a second device has nothing to run, whatever the target
    supports alone.
    """
    target = PIPELINE_REGISTRY.retrieve_architecture(TARGET)
    assert target is not None and target.multi_gpu_supported
    arch = _fused()
    assert arch.supports_device_graph_capture is False
    assert arch.multi_gpu_supported is False


def test_the_draft_arch_resolves_without_a_pipeline_model() -> None:
    """``--draft-model z-lab/Qwen3.8-27B-DFlash2`` declares
    ``DFlash2DraftModel``, which the registry must resolve before the pairing
    does. It builds its module through its config, having no model of its
    own."""
    draft = PIPELINE_REGISTRY.retrieve_architecture(DRAFT)
    assert draft is not None
    assert draft.pipeline_model is None
    assert draft.config is DFlash2Qwen3_5ArchConfig


def test_dflash2_without_a_draft_model_does_not_pair() -> None:
    assert spec_decode_architecture(TARGET, "dflash2", None) is None


def test_a_v1_dflash_draft_does_not_select_the_v2_arch() -> None:
    """v1 and v2 share a fused-graph shape but not a drafter body."""
    arch = spec_decode_architecture(TARGET, "dflash", "DFlashDraftModel")
    assert arch is None or arch.name != _fused().name
