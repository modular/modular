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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces import ModelInputs, ModelOutputs
from max.pipelines.lib.pipeline_variants import text_generation


class _Model(text_generation.PipelineModelWithKVCache[TextContext]):
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise AssertionError("Model execution is outside this test")


class _SamplerReached(Exception):
    pass


@pytest.mark.parametrize("sample_on_host", [False, True])
def test_sampler_extensions(sample_on_host: bool) -> None:
    config = MagicMock()
    config.sampling.sample_on_host = sample_on_host
    # MagicMock attributes are truthy: export-only would return before the
    # samplers, and eager would reject the mocked model.
    config.export_only = False
    config.runtime.eager = False
    device = MagicMock(label="gpu", id=0, is_host=False)
    extensions = [Path("custom_kernels")]
    model = MagicMock(sampler_custom_extensions=extensions)

    # Skip device initialization and model loading, but exercise the pipeline's
    # constructor through its sampler compilation boundary.
    with (
        patch.object(text_generation, "load_devices", return_value=[device]),
        patch.object(text_generation, "InferenceSession"),
        patch.object(text_generation, "BatchLabelScorer"),
        patch.object(text_generation, "load_weights"),
        patch.object(text_generation, "weights_format"),
        patch.object(text_generation, "load_kv_manager"),
        patch.object(text_generation.StructuredOutputHelper, "from_tokenizer"),
        patch.object(text_generation, "eager_usage_validator"),
        patch.object(
            _Model,
            "__new__",
            return_value=model,
        ),
        patch.object(
            text_generation,
            "load_sampler_graphs",
            side_effect=_SamplerReached,
        ) as load_sampler,
    ):
        with pytest.raises(_SamplerReached):
            text_generation.TextGenerationPipeline(
                pipeline_config=config,
                pipeline_model=_Model,
                weight_adapters={},
                tokenizer=MagicMock(),
                memory_plan=MagicMock(),
            )

    assert load_sampler.call_args.kwargs["custom_extensions"] == (
        () if sample_on_host else extensions
    )
