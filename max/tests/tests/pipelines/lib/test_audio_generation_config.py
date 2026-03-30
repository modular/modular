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
"""Tests for AudioGenerationConfig."""

from max.pipelines.lib import AudioGenerationConfig, PipelineRuntimeConfig


def test_audio_generation_config_field_descriptions() -> None:
    """Ensure AudioGenerationConfig exposes field descriptions for CLI help."""
    assert (
        AudioGenerationConfig.model_fields["audio_decoder"].description
        == "The name of the audio decoder model architecture."
    )


def test_audio_generation_config_disables_device_graph_capture() -> None:
    """Audio pipelines should not carry device graph capture into serve startup."""
    config = AudioGenerationConfig.model_construct(
        runtime=PipelineRuntimeConfig.model_construct(
            device_graph_capture=True
        ),
    )

    config._validate_and_resolve_overlap_scheduler()

    assert config.runtime.device_graph_capture is False
