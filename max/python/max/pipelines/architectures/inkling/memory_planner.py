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
"""Memory planner for the Inkling architecture.

A forward's scratch scales with the tokens in that forward. Each decoder
layer is one subgraph called once per layer, its per-token tensors are sized
by the ragged token dim, and the buffer plan reuses across calls, so the peak
is about one layer body's per-token footprint. No per-forward tensor of any
size is indexed by the cache length, so max_length does not enter.
"""

from __future__ import annotations

import itertools
import logging

from max.dtype import DType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

from .layers.vision import plan_folds
from .model_config import InklingTextConfig, InklingVisionConfig

logger = logging.getLogger("max.pipelines")

_ACT_DTYPE = DType.bfloat16
"""Activations are bfloat16 regardless of weight packing."""

_LOGITS_DTYPE = DType.float32

_PATCH_DTYPE = DType.float32
"""The image processor emits float32 patches."""

_HIDDEN_WIDTHS_PER_FORWARD_TOKEN = 96
"""Hidden-width activations one layer body holds per token, on one device.

Every wide per-token tensor in a body is a multiple of hidden_size, so the
count is a property of the layer code and carries across Inkling sizes. It
is a census of the .mo dump's tensor<[total_seq_len, ...]> types (85 on
Inkling Small at TP=2, 81 to 92 measured on device), rounded up. A change to
the decoder or its fusion moves it, so re-census rather than trust it.
"""

_UNRESOLVED_BATCH_SIZE = 512
"""Logit rows to price when max_batch_size is still unset on the config.

Matches the framework default so the guess is not below what planning picks.
"""


def _peak_vision_bytes_per_patch(vision_config: InklingVisionConfig) -> int:
    """Peak bytes one image patch occupies in the tower, on one device.

    The tower folds a patch into the decoder width one step at a time and
    holds a step's input and output together.
    """
    act = _ACT_DTYPE.size_in_bytes
    t = vision_config.temporal_patch_size
    h = w = vision_config.patch_size
    elements = [t * h * w * vision_config.n_channels]
    for t_fold, hw_fold, out_dim in plan_folds(vision_config):
        t //= t_fold
        h //= hw_fold
        w //= hw_fold
        elements.append(t * h * w * out_dim)
    # The float32 input is live into the first cast.
    return max(
        elements[0] * _PATCH_DTYPE.size_in_bytes + elements[1] * act,
        max(
            (
                (before + after) * act
                for before, after in itertools.pairwise(elements[1:])
            ),
            default=0,
        ),
    )


class InklingMemoryPlanner(PagedMemoryPlanner):
    """Memory planner for Inkling: a MoE decoder behind a vision tower."""

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserves what the widest forward allocates, across every device.

        Scales with the tokens in one forward: max_batch_input_tokens under
        chunked prefill, the whole prompt without it. The conv state is pages
        of the KV pool and the MTP draft's scratch has its own planner.
        """
        text_config = InklingTextConfig.from_hf(huggingface_config.text_config)
        vision_config = InklingVisionConfig.from_hf(
            huggingface_config.vision_config
        )
        n_devices = len(pipeline_config.model.device_specs)
        max_length = pipeline_config.model.max_length
        assert max_length is not None
        rows = max(
            pipeline_config.runtime.max_batch_input_tokens
            if pipeline_config.runtime.enable_chunked_prefill
            else max_length,
            pipeline_config.runtime.max_batch_size or 1,
        )

        decoder = (
            _HIDDEN_WIDTHS_PER_FORWARD_TOKEN
            * text_config.hidden_size
            * _ACT_DTYPE.size_in_bytes
            * rows
            * n_devices
        )

        # One patch is one decoder token. The tower runs on one device.
        vision = _peak_vision_bytes_per_patch(vision_config) * rows

        # Only sampled rows reach the vocabulary, on one device.
        logits = (
            min(
                pipeline_config.runtime.max_batch_size
                or _UNRESOLVED_BATCH_SIZE,
                rows,
            )
            * text_config.vocab_size
            * _LOGITS_DTYPE.size_in_bytes
        )

        total = decoder + vision + logits
        logger.info(
            "Estimated activation memory: %s (decoder=%s over %d tokens per"
            " forward, vision=%s, logits=%s)",
            to_human_readable_bytes(total),
            to_human_readable_bytes(decoder),
            rows,
            to_human_readable_bytes(vision),
            to_human_readable_bytes(logits),
        )
        return total
