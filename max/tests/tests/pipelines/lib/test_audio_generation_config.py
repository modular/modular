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

from max.pipelines.lib import AudioGenerationConfig


def test_audio_generation_config_field_descriptions() -> None:
    """Ensure AudioGenerationConfig exposes field descriptions for CLI help."""
    assert (
        AudioGenerationConfig.model_fields["audio_decoder"].description
        == "The name of the audio decoder model architecture."
    )
