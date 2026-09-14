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
"""Model inputs for the Kimi-K2.5 ModuleV3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class KimiK2_5ModelInputs(ModelInputs):
    """Flat ModuleV3 inputs for the Kimi-K2.5 model.

    The language ABI is ``(tokens, return_n_logits, input_row_offsets,
    vision_embeddings, vision_scatter_indices, *kv, *ep)`` — the DeepseekV3
    ModuleV3 order with the two multimodal tensors spliced in after the row
    offsets. ``vision_embeddings``/``vision_scatter_indices`` are the base
    :class:`ModelInputs` fields, set by the pipeline's vision seam
    (``finalize_vision_inputs``); replicated per device (one ``Buffer`` per
    device, identical data). Shape ``[num_patches, hidden]`` /
    ``[num_image_tokens]`` during prefill, ``[0, hidden]`` / ``[0]`` otherwise.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    batch_context_lengths: list[Buffer] = field(kw_only=True)
    """Host (CPU) page-aligned KV context length, one per DP replica.

    Substituted for the planner's device-resident ``buffer_lengths`` so the
    per-layer ``.to(CPU())`` stays host-to-host and the graph is capturable."""

    data_parallel_splits: Buffer | None = field(default=None, kw_only=True)
    input_row_offsets_i64: Buffer | None = field(default=None, kw_only=True)
    ep_inputs: tuple[Buffer, ...] = field(default=(), kw_only=True)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Flat language-model input tuple in compile ABI order."""
        dp_inputs: tuple[Buffer, ...] = ()
        if self.data_parallel_splits is not None:
            assert self.input_row_offsets_i64 is not None
            dp_inputs = (self.data_parallel_splits, self.input_row_offsets_i64)
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            *self.batch_context_lengths,
            *dp_inputs,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.ep_inputs,
        )
