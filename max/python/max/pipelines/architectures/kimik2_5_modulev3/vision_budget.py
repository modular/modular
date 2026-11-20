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

"""Vision encoder token budgeting for Kimi K2.5."""

from __future__ import annotations

from max.pipelines.lib.config import PipelineConfig

from .model_config import VisionConfig


def vision_encoder_token_budget(
    pipeline_config: PipelineConfig,
) -> int | None:
    """Return the per-call image-token ceiling for the vision encoder.

    Returns ``max_batch_input_tokens`` so the vision encoder respects the
    same input-token budget the LM forward pass honors under chunked prefill.
    Returns ``None`` when ``max_batch_input_tokens`` is unset, signalling
    that chunking should be disabled.
    """
    max_batch_input_tokens = int(
        pipeline_config.runtime.max_batch_input_tokens or 0
    )
    if max_batch_input_tokens <= 0:
        return None
    return max_batch_input_tokens


def vision_merge_sq(vision_config: VisionConfig) -> int:
    """Patches-per-output-token from the vision config's merge kernel."""
    merge_kernel_size = vision_config.merge_kernel_size
    return max(1, merge_kernel_size[0] * merge_kernel_size[1])
