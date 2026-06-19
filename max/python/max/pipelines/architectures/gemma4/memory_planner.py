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

# Tokens processed in a single forward step for the activation estimate below.
# Decode runs ``max_batch_size`` tokens/step; chunked prefill runs up to this
# many. Used to right-size the reservation for low-concurrency serving (e.g. a
# single NVFP4 checkpoint on a 24 GB pre-Blackwell card) without ever exceeding
# the previous flat reservation.
_PREFILL_TOKENS_PER_STEP = 8192

# Safety multiple over the widest per-token transient (hidden vs MLP
# intermediate), covering the handful of simultaneously-live layer activations,
# attention temporaries and the LM head.
# TODO(MODELS-1544): calibrate against measured peak GPU memory (A10G/L40S NVFP4
# run) before relaxing the ``min(..., flat)`` cap in estimate_activation_memory.
_ACTIVATION_SAFETY_FACTOR = 8


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
        """Estimates activation memory, scaled by model dimensions and capped at
        the previous flat reservation (so it never reserves more than before).

        The scaled value is an uncalibrated heuristic, not a proven
        activation-peak bound, so it can under-reserve below the cap
        (MODELS-1544).

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: Provides ``hidden_size`` / ``intermediate_size``.

        Returns:
            Estimated activation memory in bytes, summed across all devices.
        """
        num_devices = len(pipeline_config.model.device_specs)

        # Previous behaviour, kept as a strict upper bound. Smaller KV cache
        # dtypes (e.g. FP8) buy ~2x more blocks, so the scheduler targets larger
        # concurrent batches whose activations need proportionally more headroom.
        flat = (
            30 // pipeline_config.model.kv_cache.cache_dtype.size_in_bytes
        ) * 1024**3

        # Peak transient activations scale with the widest per-token buffer, the
        # tokens processed in one forward step, and a safety multiple. The
        # widest buffer is the MLP intermediate (or hidden, whichever is larger).
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        width = max(
            getattr(text_config, "hidden_size", 0),
            getattr(text_config, "intermediate_size", 0),
        )
        if width <= 0:
            # Dimensions unavailable: fall back to the conservative flat value.
            base = flat
        else:
            max_batch = pipeline_config.runtime.max_batch_size or 1
            tokens_per_step = max(max_batch, _PREFILL_TOKENS_PER_STEP)
            # Activations compute in the dequantized model dtype; NVFP4 and bf16
            # checkpoints both compute in bf16 (2 bytes).
            principled = _ACTIVATION_SAFETY_FACTOR * tokens_per_step * width * 2
            base = min(principled, flat)

        if pipeline_config.runtime.device_graph_capture:
            base += _GRAPH_CAPTURE_HEADROOM_BYTES
        return base * num_devices

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
