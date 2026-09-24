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
"""Input batching for Inkling pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.engine import Model
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import TextAndVisionContext
from max.pipelines.graph_input_stager import InputDescriptor
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    RAGGED_INPUT_ROW_OFFSETS,
    RAGGED_INPUT_TOKENS,
    BatchProcessorRuntime,
    SingleReplicaRaggedBatchProcessor,
    single_replica_context_batch,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelInputs
from max.pipelines.lib.vision_batching import (
    create_empty_image_embeddings_single,
    create_empty_image_token_indices_single,
)
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices

from .model_config import InklingConfig

_TOKEN_POSITIONS = "token_positions"
"""Device input holding each token's position within its own sequence."""


@dataclass
class InklingInputs(ModelInputs):
    """Ragged token inputs plus the convolution-state pool addressing."""

    tokens: Buffer
    input_row_offsets: Buffer
    positions: Buffer
    return_n_logits: Buffer

    image_embeddings: Buffer
    image_indices: Buffer
    """Token-stream row each vision row replaces; negative entries are skipped."""

    signal_buffers: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.positions,
            self.return_n_logits,
            self.image_embeddings,
            self.image_indices,
            *self.signal_buffers,
            # The conv state rides inside the cache tree, so one walk
            # covers the attention pages and the state leaves both.
            *tree.leaves(self.kv_cache_inputs),
        )


class InklingBatchProcessor(
    SingleReplicaRaggedBatchProcessor[TextAndVisionContext, InklingInputs]
):
    """Ragged batching with Inkling's vision operands.

    The cache manager claims each request's conv-state row before this runs.
    """

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(
            config,
            runtime,
            # One position per token, so the token stream's bound covers it.
            [
                InputDescriptor(
                    name=_TOKEN_POSITIONS,
                    dtype=DType.uint32,
                    max_shape=(runtime.max_batch_active_tokens,),
                    destinations=[runtime.devices[0]],
                )
            ],
        )
        assert isinstance(config, InklingConfig)
        self._hidden_size = config.text_config.hidden_size
        self._dtype = config.dtype
        self._vision_model: Model | None = None
        self._signal_buffers = list(runtime.signal_buffers)
        self._return_n_logits_buffers: dict[int, Buffer] = {}
        self._no_images: tuple[Buffer, Buffer] | None = None

    def bind_runtime_state(self, vision_model: Model) -> None:
        """Hands over what only exists once the model is compiled and loaded."""
        self._vision_model = vision_model

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InklingInputs:
        context_batch = single_replica_context_batch(
            replica_batches, processor_name=type(self).__qualname__
        )
        tokens, input_row_offsets, positions = self._stage_token_inputs(
            context_batch
        )

        # Reusing one buffer per distinct value keeps graph-capture replay
        # from recopying an input that never changes.
        return_n_logits_buffer = self._return_n_logits_buffers.get(
            return_n_logits
        )
        if return_n_logits_buffer is None:
            return_n_logits_buffer = Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            )
            self._return_n_logits_buffers[return_n_logits] = (
                return_n_logits_buffer
            )

        image_embeddings, image_indices = self._image_operands(context_batch)

        return self._make_inkling_inputs(
            context_batch=context_batch,
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            positions=positions,
            return_n_logits=return_n_logits_buffer,
            image_embeddings=image_embeddings,
            image_indices=image_indices,
            signal_buffers=self._signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def _make_inkling_inputs(
        self,
        *,
        context_batch: Sequence[TextAndVisionContext],
        tokens: Buffer,
        input_row_offsets: Buffer,
        positions: Buffer,
        return_n_logits: Buffer,
        image_embeddings: Buffer,
        image_indices: Buffer,
        signal_buffers: list[Buffer],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
    ) -> InklingInputs:
        """Constructs this processor's ``*Inputs`` from the batched fields."""
        return InklingInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            positions=positions,
            return_n_logits=return_n_logits,
            image_embeddings=image_embeddings,
            image_indices=image_indices,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def _stage_token_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> tuple[Buffer, Buffer, Buffer]:
        lengths = np.fromiter(
            (context.tokens.active_length for context in context_batch),
            dtype=np.int64,
            count=len(context_batch),
        )
        total_seq_len = int(lengths.sum())

        with self._stager.stage() as staging:
            host_tokens, (device_tokens,) = staging.get(
                RAGGED_INPUT_TOKENS, (total_seq_len,)
            )
            host_row_offsets, (device_row_offsets,) = staging.get(
                RAGGED_INPUT_ROW_OFFSETS, (len(context_batch) + 1,)
            )
            host_positions, (device_positions,) = staging.get(
                _TOKEN_POSITIONS, (total_seq_len,)
            )

            offsets = np.cumsum([0, *lengths], dtype=np.int64)
            host_row_offsets.to_numpy()[:] = offsets
            if total_seq_len:
                np.concatenate(
                    [context.tokens.active for context in context_batch],
                    out=host_tokens.to_numpy(),
                )
            # One ramp shifted per sequence covers the whole ragged batch.
            first_positions = (
                np.fromiter(
                    (
                        context.tokens.current_position
                        for context in context_batch
                    ),
                    dtype=np.int64,
                    count=len(context_batch),
                )
                - lengths
            )
            host_positions.to_numpy()[:] = np.arange(
                total_seq_len, dtype=np.int64
            ) + np.repeat(first_positions - offsets[:-1], lengths)

        return device_tokens, device_row_offsets, device_positions

    def _empty_image_operands(self) -> tuple[Buffer, Buffer]:
        """Zero-row operands for a batch with no images to encode."""
        if self._no_images is None:
            device0 = self.runtime.devices[0]
            self._no_images = (
                create_empty_image_embeddings_single(
                    device0, self._hidden_size, self._dtype
                ),
                create_empty_image_token_indices_single(device0),
            )
        return self._no_images

    def _image_operands(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> tuple[Buffer, Buffer]:
        """Vision-tower rows for this batch, and the rows they replace."""
        # The whole batch feeds the helper, so the row offsets it returns cover
        # the ragged token stream. A graph-capture warmup probe batches bare
        # TextContexts, which is why the attribute is read defensively.
        indices = compute_multimodal_merge_indices(context_batch)
        if indices.size == 0:
            return self._empty_image_operands()
        blocks = [
            image.pixel_values
            for context in context_batch
            if getattr(context, "needs_vision_encoding", False)
            for image in context.images
        ]
        assert self._vision_model is not None
        device0 = self.runtime.devices[0]
        embeddings = self._vision_model.execute(
            Buffer.from_numpy(np.concatenate(blocks)).to(device0)
        )[0]
        assert isinstance(embeddings, Buffer)
        return embeddings, Buffer.from_numpy(indices).to(device0)
