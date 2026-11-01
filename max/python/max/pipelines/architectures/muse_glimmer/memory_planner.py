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

"""Memory planner for the Muse Glimmer architecture."""

from __future__ import annotations

from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from transformers import AutoConfig

from .model_config import MuseGlimmerConfig


class MuseGlimmerMemoryPlanner(PagedMemoryPlanner):
    """Reserves a fixed activation budget, sized from the KV cache dtype."""

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates activation memory for Muse Glimmer models.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: Unused.

        Returns:
            Estimated activation memory in bytes, summed across all devices.
        """
        # TODO: this is Gemma 4's budget, not measured for this model's text
        # and vision activations; measure it before raising the batch size.
        # Smaller KV cache dtypes buy more blocks, so the scheduler admits
        # larger batches whose activations need proportionally more room.
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, MuseGlimmerConfig.DEFAULT_ENCODING
        )
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )
        base = (30 // cache_dtype.size_in_bytes) * 1024**3
        return base * len(pipeline_config.model.device_specs)
