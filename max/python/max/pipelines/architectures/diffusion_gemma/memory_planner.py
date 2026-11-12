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
"""Memory planner for the DiffusionGemma architecture."""

from __future__ import annotations

from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from transformers import AutoConfig

from .model_config import Gemma4ForConditionalGenerationConfig


class DiffusionGemmaMemoryPlanner(PagedMemoryPlanner):
    """Plans the KV cache around both text graphs.

    The encoder and the denoiser compile as two graphs over one shared
    weight set, and each graph holds its own device copy of the text
    weights.
    """

    _always_signal_buffers = True

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Returns the checkpoint's weights plus the second text-graph copy.

        It over-reserves by the vision tower's share, which only one graph
        holds.
        """
        return 2 * super().estimate_weights_size(pipeline_config)

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Returns the activation headroom, summed across devices.

        The headroom is a base sized from the KV cache dtype, scaled by the
        device count to match the total-across-devices budget in
        :meth:`MemoryEstimator.plan_from_sizes`. A smaller KV cache dtype
        (e.g. FP8) halves bytes per block, so the same budget buys about
        twice the blocks, and the scheduler admits larger concurrent
        batches whose tensors need proportionally more room. With a BF16 KV
        cache it is 15 GiB, which holds the denoise step at the
        block-diffusion pipeline's 16 canvases: about 11 GiB of
        activations and the 2 GiB of self-conditioning logits it reads.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: Unused.

        Returns:
            Estimated activation memory in bytes, summed across all devices.
        """
        # FIXME: the headroom is an arbitrary guess at how much the vision
        # path and the denoising step need.
        # TODO(MODELS-1544): investigate high activation memory estimates.
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model,
            Gemma4ForConditionalGenerationConfig.DEFAULT_ENCODING,
        )
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )
        headroom = (30 // cache_dtype.size_in_bytes) * 1024**3
        return headroom * len(pipeline_config.model.device_specs)
