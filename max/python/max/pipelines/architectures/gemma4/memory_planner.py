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

"""Memory planner for the Gemma4 architecture."""

from __future__ import annotations

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from transformers import AutoConfig

_GRAPH_CAPTURE_HEADROOM_BYTES = 2 * 1024**3


class Gemma4MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for Gemma4 (vision-language) models.

    Reserves a per-device activation budget (a base sized from the KV cache
    dtype, plus optional graph-capture headroom), scaled by the device count to
    match the total-across-devices budget in
    :meth:`MemoryEstimator.estimate_memory_footprint`.  Also provides vision
    cache entry byte estimation for the KV-and-vision-cache reservation path.
    """

    _always_signal_buffers = True

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates activation memory for Gemma4 models.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: Unused.

        Returns:
            Estimated activation memory in bytes, summed across all devices.
        """
        # FIXME: We arbitrarily set some memory for activation memory to leave
        # headroom for vision processing. We should determine this in a more
        # principled way.
        # Smaller KV cache dtypes (e.g. FP8) halve bytes_per_block, so the
        # same KV budget buys ~2x more blocks.  The scheduler admits work
        # based on available blocks, so it targets larger concurrent batches
        # whose activation tensors need proportionally more headroom.
        # TODO(MODELS-1544): investigate high activation memory estimates
        base = (
            30 // pipeline_config.model.kv_cache.cache_dtype.size_in_bytes
        ) * 1024**3
        if pipeline_config.runtime.device_graph_capture:
            base += _GRAPH_CAPTURE_HEADROOM_BYTES
        return base * len(pipeline_config.model.device_specs)

    def estimate_vision_cache_entry_bytes(
        self,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates per-entry bytes for the Gemma4 vision encoder cache.

        Worst-case tokens per image is
        ``position_embedding_size / pooling_kernel_size²``, stored at the text
        hidden size in bfloat16.

        Args:
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated bytes per vision cache entry.

        Raises:
            ValueError: If the required vision or text config is absent.
        """
        vision_config = getattr(huggingface_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(
                "Gemma4 requires a vision_config in the HuggingFace config"
            )
        text_config = getattr(huggingface_config, "text_config", None)
        if text_config is None:
            raise ValueError(
                "Gemma4 requires a text_config in the HuggingFace config"
            )
        if getattr(huggingface_config, "model_type", None) == "gemma4_unified":
            # These checkpoints are served text-only (different vision
            # schema); no vision cache is needed.
            return 0
        k = vision_config.pooling_kernel_size
        max_tokens = vision_config.position_embedding_size // (k * k)
        hidden = text_config.hidden_size
        return max_tokens * hidden * 2  # bfloat16
