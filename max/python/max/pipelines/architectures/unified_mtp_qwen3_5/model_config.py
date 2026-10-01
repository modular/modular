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
"""The fused Qwen3.5 MTP graph's config: what it reads, allocates and prices."""

from __future__ import annotations

from max.dtype import DType
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from ..qwen3_5.model_config import Qwen3_5Config
from .unified_mtp_qwen3_5 import ring_len_for_config


class UnifiedMTPQwen3_5Config(Qwen3_5Config):
    """Qwen3.5's config with the vision-cache facts withdrawn.

    The fused MTP graph is text-only -- ``_create_model_config`` drops
    ``vision_config`` so no encoder is compiled -- while the checkpoint it
    reads still declares a vision tower. Reporting the base architecture's
    per-entry estimate would make memory planning reserve a slice of the KV
    pool for an encoder cache this graph can never fill.

    The state cache also holds each request's verify ring, as a scratch
    leaf.
    """

    @classmethod
    def _verify_ring_len(cls, pipeline_config: PipelineConfig) -> int:
        return ring_len_for_config(pipeline_config.speculative)

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        return 0

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        return None
