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
"""Tests for PipelineRuntimeConfig."""

from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig


def test_emit_reasoning_content_defaults_false() -> None:
    assert PipelineRuntimeConfig().emit_reasoning_content is False


def test_emit_reasoning_content_can_be_enabled() -> None:
    assert (
        PipelineRuntimeConfig(
            emit_reasoning_content=True
        ).emit_reasoning_content
        is True
    )
