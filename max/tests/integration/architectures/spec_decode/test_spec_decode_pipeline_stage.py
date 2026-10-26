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

"""Which pipeline stages a fused speculative graph builds, and what a later
stage takes beside the unstaged signature.

Only a pairing whose target adapter opts in speculates on a stage, and only
on the last of two or more, where the draft finds everything it reads.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

import pytest
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.pipelines.lib.pipeline_parallel import PipelinedModel, pipeline_stages
from max.pipelines.lib.pipeline_stage import PipelineStage
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    _stage_boundary_input_types,
    _stage_error,
)
from max.pipelines.speculative import spec_adapters

GEMMA4 = "Gemma4ForConditionalGeneration"
GEMMA4_ASSISTANT = "Gemma4AssistantForCausalLM"
LLAMA = "LlamaForCausalLM"


def _config(
    target: str, method: str, draft: str, stage: PipelineStage | None
) -> Any:
    """Stands in for a resolved pipeline config of a pairing at ``stage``."""
    return SimpleNamespace(
        models=SimpleNamespace(main_architecture_name=target),
        speculative=SimpleNamespace(speculative_method=method),
        draft_model=SimpleNamespace(
            huggingface_config=SimpleNamespace(architectures=[draft])
        ),
        model=SimpleNamespace(pipeline_stage=stage),
    )


def test_the_fused_model_builds_pipeline_stages() -> None:
    assert issubclass(SpecDecodePipelineModel, PipelinedModel)


def test_only_gemma4_mtp_speculates_on_a_pipeline_stage() -> None:
    assert spec_adapters.speculates_on_last_stage(GEMMA4, "mtp")
    assert not spec_adapters.speculates_on_last_stage(GEMMA4, "dflash")
    assert not spec_adapters.speculates_on_last_stage(LLAMA, "eagle")


def test_gemma4_mtp_builds_a_later_last_stage() -> None:
    last = pipeline_stages([30, 30])[1]
    config = _config(GEMMA4, "mtp", GEMMA4_ASSISTANT, last)
    assert _stage_error(config) is None
    assert SpecDecodePipelineModel.stage_speculation_error(config) is None


@pytest.mark.parametrize(
    "sizes, index",
    [
        pytest.param([30, 30], 0, id="first of two"),
        pytest.param([20, 20, 20], 1, id="middle of three"),
        pytest.param([60], 0, id="unsplit stage"),
    ],
)
def test_no_stage_but_a_later_last_one_speculates(
    sizes: list[int], index: int
) -> None:
    stage = pipeline_stages(sizes)[index]
    error = _stage_error(_config(GEMMA4, "mtp", GEMMA4_ASSISTANT, stage))
    assert error and "only the last of two or more" in error, error


def test_a_pairing_that_does_not_opt_in_is_refused_on_a_stage() -> None:
    last = pipeline_stages([16, 16])[1]
    config = _config(LLAMA, "eagle", LLAMA, last)
    error = SpecDecodePipelineModel.stage_speculation_error(config)
    assert error and "does not speculate with eagle" in error, error
    assert _stage_error(config) == error


def test_the_whole_model_builds_unstaged() -> None:
    assert _stage_error(_config(LLAMA, "eagle", LLAMA, None)) is None


def test_a_later_stage_takes_its_boundary_over_the_merged_window() -> None:
    hidden = TensorType(
        DType.bfloat16, ["total_seq_len", 64], device=DeviceRef.GPU()
    )
    first, last = pipeline_stages([2, 2])
    later = PipelineStage(
        index=last.index,
        num_stages=last.num_stages,
        layers=last.layers,
        boundary_types=OrderedDict(hidden_states=hidden),
    )
    assert _stage_boundary_input_types(None) == []
    assert _stage_boundary_input_types(first) == []
    (boundary,) = _stage_boundary_input_types(later)
    assert boundary.dtype == DType.bfloat16
    assert [str(dim) for dim in boundary.shape] == ["merged_seq_len", "64"]
    assert boundary.device == hidden.device
