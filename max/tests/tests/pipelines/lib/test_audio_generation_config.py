# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


def test_audio_generation_config_missing_help_method() -> None:
    """Test that AudioGenerationConfig is missing a help() method and should have one."""

    # Check if AudioGenerationConfig has its own help method or inherits from PipelineConfig
    assert "help" in AudioGenerationConfig.__dict__, (
        "AudioGenerationConfig should have its own help() method"
    )
