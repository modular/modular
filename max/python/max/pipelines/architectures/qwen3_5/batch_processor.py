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
"""Input batching for Qwen3.5 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
    VisionEncodingData,
)
from max.pipelines.context import TextContext
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices

if TYPE_CHECKING:
    from .model import Qwen3_5Inputs
    from .state_cache import GatedDeltaNetStateCache


class Qwen3_5BatchProcessor(Llama3BatchProcessor):
    """Ragged batching with linear-attention state pools and optional vision inputs."""

    _state_cache: GatedDeltaNetStateCache | None = None
    _slot_idx_prealloc: Buffer | None = None
    _empty_lm_image_embeddings: Buffer | None = None
    _empty_lm_image_token_indices: Buffer | None = None

    def bind_prepare_state(
        self,
        *,
        state_cache: GatedDeltaNetStateCache,
        slot_idx_prealloc: Buffer,
        empty_lm_image_embeddings: Buffer | None = None,
        empty_lm_image_token_indices: Buffer | None = None,
    ) -> None:
        """Wires state pools and vision placeholders created during ``load_model``."""
        self._state_cache = state_cache
        self._slot_idx_prealloc = slot_idx_prealloc
        self._empty_lm_image_embeddings = empty_lm_image_embeddings
        self._empty_lm_image_token_indices = empty_lm_image_token_indices

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3_5Inputs:
        from .model import Qwen3_5Inputs

        base_inputs = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )

        all_contexts = [ctx for batch in replica_batches for ctx in batch]
        request_ids = [ctx.request_id for ctx in all_contexts]

        assert self._state_cache is not None, (
            "Qwen3.5 always has linear-attention layers; state cache must "
            "be initialised by load_model()"
        )
        assert self._slot_idx_prealloc is not None
        for rid in request_ids:
            self._state_cache.claim(rid)
        slot_idx = self._state_cache.slot_idx_for(
            request_ids, self._slot_idx_prealloc
        )
        conv_pools = self._state_cache.conv_pools
        recurrent_pools = self._state_cache.rec_pools

        pixel_values: Buffer | None = None
        weights: Buffer | None = None
        indices: Buffer | None = None
        vision_position_ids: Buffer | None = None
        max_grid_size: Buffer | None = None
        grid_thw: Buffer | None = None
        cu_seqlens: Buffer | None = None
        max_seqlen: Buffer | None = None
        image_token_indices: Buffer | None = None

        if self._empty_lm_image_embeddings is not None:
            vision_contexts = [
                ctx
                for ctx in all_contexts
                if isinstance(ctx, Qwen3VLTextAndVisionContext)
            ]
            vision_datas: list[VisionEncodingData] = []
            for ctx in vision_contexts:
                if ctx.needs_vision_encoding:
                    assert ctx.vision_data is not None, (
                        "vision_data must be set when needs_vision_encoding is True"
                    )
                    vision_datas.append(ctx.vision_data)

            if vision_contexts:
                np_indices = compute_multimodal_merge_indices(vision_contexts)
                image_token_indices = Buffer.from_numpy(np_indices).to(
                    self.runtime.devices[0]
                )
            else:
                image_token_indices = self._empty_lm_image_token_indices

            if vision_datas:
                device0 = self.runtime.devices[0]
                pixel_values = Buffer.from_numpy(
                    np.concatenate(
                        [vd.concatenated_pixel_values for vd in vision_datas]
                    ).astype(np.float32)
                ).to(device0)

                weights = Buffer.from_numpy(
                    np.concatenate(
                        [vd.weights for vd in vision_datas], axis=1
                    ).astype(np.float32)
                ).to(device0)

                indices = Buffer.from_numpy(
                    np.concatenate([vd.indices for vd in vision_datas], axis=1)
                ).to(device0)

                vision_position_ids = Buffer.from_numpy(
                    np.concatenate(
                        [vd.vision_position_ids for vd in vision_datas]
                    ).astype(np.int32)
                ).to(device0)

                grid_thw = Buffer.from_numpy(
                    np.concatenate(
                        [vd.image_grid_thw for vd in vision_datas]
                    ).astype(np.int64)
                ).to(device0)

                max_grid_size_value = max(
                    vd.max_grid_size.item() for vd in vision_datas
                )
                max_grid_size = Buffer.from_numpy(
                    np.array(max_grid_size_value, dtype=np.int32)
                )

                cu_seqlens_list = []
                offset = np.uint32(0)
                for vd in vision_datas:
                    seqlens = vd.cu_seqlens.copy()
                    seqlens[1:] += offset
                    cu_seqlens_list.append(seqlens[1:])
                    offset = seqlens[-1]
                cu_seqlens = Buffer.from_numpy(
                    np.concatenate(
                        [np.array([0], dtype=np.uint32), *cu_seqlens_list]
                    ).astype(np.uint32)
                ).to(device0)

                max_seqlen_value = max(
                    vd.max_seqlen.item() for vd in vision_datas
                )
                max_seqlen = Buffer.from_numpy(
                    np.array([max_seqlen_value], dtype=np.uint32)
                )

        return Qwen3_5Inputs(
            tokens=base_inputs.tokens,
            input_row_offsets=base_inputs.input_row_offsets,
            signal_buffers=base_inputs.signal_buffers,
            kv_cache_inputs=base_inputs.kv_cache_inputs,
            return_n_logits=base_inputs.return_n_logits,
            slot_idx=slot_idx,
            conv_pools=conv_pools,
            recurrent_pools=recurrent_pools,
            request_ids=request_ids,
            lm_image_embeddings=self._empty_lm_image_embeddings,
            image_token_indices=image_token_indices,
            pixel_values=pixel_values,
            vision_position_ids=vision_position_ids,
            weights=weights,
            indices=indices,
            max_grid_size=max_grid_size,
            grid_thw=grid_thw,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
