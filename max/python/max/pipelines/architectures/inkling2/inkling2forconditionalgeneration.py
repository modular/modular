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

"""The decoder and the vision tower under one module tree."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from max.nn import Module
from max.tensor import Tensor

from .inkling import Inkling
from .layers.vision import InklingVisionModel
from .model_config import Inkling2Config


class Inkling2ForConditionalGeneration(Module[..., Any]):
    """The decoder and the tower under one module tree."""

    # The three graph inputs that are neither tokens/offsets nor cache leaves,
    # as (dtype, shape), in ``forward``'s order. The decoder names them in its
    # signature, so they sit between ``input_row_offsets`` and the cache
    # leaves rather than trailing them.
    EXTRA_INPUT_TYPES: ClassVar[dict[str, tuple[str, list[str]]]] = {
        # The absolute position of every token in this step
        # (`decoder_input_types` declares the same dtype for this field).
        "positions": ("uint32", ["total_seq_len"]),
        # One row per image token, and the row each lands on. A text-only step
        # carries zero rows, which the merge skips.
        "image_embeddings": ("float32", ["num_image_tokens", "hidden_size"]),
        "image_indices": ("int32", ["total_image_tokens"]),
    }
    # This step's VALUES for those extras. The image width is the checkpoint's,
    # so the builders are bound in __init__.
    EXTRA_INPUT_BUILDERS: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        config: Inkling2Config,
        kv_params: object | None = None,
        **kwargs: object,
    ) -> None:
        # kv_params: accepted and ignored. The config already carries the KV
        # tree.
        del kv_params
        self.language_model = Inkling(config)
        self.vision = InklingVisionModel(
            config.vision_config,
            config.dtype,
        )
        width = config.text_config.hidden_size
        type(self).EXTRA_INPUT_BUILDERS = {
            # `ctx_ids` is the whole row fed so far and `step_tokens` this
            # step's slice of it, so the ramp starts at the cached length.
            "positions": lambda *, step_tokens, ctx_ids: np.arange(
                len(ctx_ids) - len(step_tokens), len(ctx_ids), dtype=np.uint32
            ),
            "image_embeddings": lambda *, step_tokens, ctx_ids: np.zeros(
                (0, width), dtype=np.float32
            ),
            "image_indices": lambda *, step_tokens, ctx_ids: np.zeros(
                (0,), dtype=np.int32
            ),
        }

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        positions: Tensor,
        image_embeddings: Tensor,
        image_indices: Tensor,
        *kv_inputs: Tensor,
    ) -> Any:
        """Runs the decoder; the tower is invoked through its own graph."""
        return self.language_model(
            tokens,
            return_n_logits,
            input_row_offsets,
            positions,
            image_embeddings,
            image_indices,
            *kv_inputs,
        )
