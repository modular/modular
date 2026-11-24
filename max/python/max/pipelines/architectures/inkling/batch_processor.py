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

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import Buffer, Device, Usage
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.input_binding import _flatten_role_inputs
from max.pipelines.lib.interfaces.batch_processor import (
    RAGGED_INPUT_ROW_OFFSETS,
    RAGGED_INPUT_TOKENS,
    BatchProcessorRuntime,
    SingleReplicaRaggedBatchProcessor,
    single_replica_context_batch,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelInputs
from max.pipelines.lib.registration_graph_api import _decoder_input_fields
from max.pipelines.lib.role_contracts import DecoderInputs, VisionInputs
from max.pipelines.lib.vision_batching import (
    create_empty_image_embeddings_single,
    create_empty_image_token_indices_single,
)
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices

from .model_config import InklingConfig


@dataclass
class InklingInputs(ModelInputs):
    """Ragged token inputs plus the convolution-state pool addressing."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    image_embeddings: Buffer
    image_indices: Buffer
    """Token-stream row each vision row replaces; negative entries are skipped."""

    signal_buffers: list[Buffer]
    input_tree_def: tree.TreeDef | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, in the graph's input order."""
        assert self.kv_cache_inputs is not None
        if self.input_tree_def is not None:
            inputs: DecoderInputs[tree.Tree[Buffer], Buffer] = DecoderInputs(
                tokens=self.tokens,
                input_row_offsets=self.input_row_offsets,
                return_n_logits=self.return_n_logits,
                vision=VisionInputs(
                    embeddings=(self.image_embeddings,),
                    scatter_indices=(self.image_indices,),
                ),
                signals=tuple(self.signal_buffers),
                kv=self.kv_cache_inputs,
            )
            values = _flatten_role_inputs(
                self.input_tree_def, inputs, strict=True
            )
            buffers: list[Buffer] = []
            for value in values:
                assert isinstance(value, Buffer)
                buffers.append(value)
            return tuple(buffers)
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            self.image_embeddings,
            self.image_indices,
            *self.signal_buffers,
            # The conv state rides inside the cache tree, so one walk
            # covers the attention pages and the state leaves both.
            *tree.leaves(self.kv_cache_inputs),
        )


class InklingBatchProcessor(
    SingleReplicaRaggedBatchProcessor[
        TextAndVisionContext, InklingInputs, InklingConfig
    ]
):
    """Ragged batching with Inkling's vision operands.

    The cache manager claims each request's conv-state row before this runs.
    """

    def __init__(
        self, config: InklingConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._hidden_size = config.text_config.hidden_size
        self._dtype = config.dtype
        self._vision_model: Model | None = None
        self._signal_buffers = list(runtime.signal_buffers)
        self._return_n_logits_buffers: dict[int, Buffer] = {}
        self._no_images: tuple[Buffer, Buffer] | None = None

    def bind_vision_encoder(
        self, *, vision_model: Model, session: InferenceSession | None = None
    ) -> None:
        """Wires the compiled vision tower, which encodes each batch's new images."""
        del session
        self._vision_model = vision_model

    def bind_input_tree(self, input_tree: OrderedDict[str, object]) -> None:
        """Binds the same semantic tree the decoder exports."""
        super().bind_input_tree(input_tree)
        self._input_tree_def = tree.flatten(_decoder_input_fields(input_tree))[
            1
        ]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InklingInputs:
        """Builds the first step's inputs from the replica batches."""
        context_batch = single_replica_context_batch(
            replica_batches, processor_name=type(self).__qualname__
        )
        tokens, input_row_offsets = self._stage_token_inputs(context_batch)

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
            return_n_logits=return_n_logits,
            image_embeddings=image_embeddings,
            image_indices=image_indices,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            input_tree_def=getattr(self, "_input_tree_def", None),
        )

    def _stage_token_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> tuple[Buffer, Buffer]:
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
            host_row_offsets.to_numpy()[:] = np.cumsum(
                [0, *lengths], dtype=np.int64
            )
            if total_seq_len:
                np.concatenate(
                    [context.tokens.active for context in context_batch],
                    out=host_tokens.to_numpy(),
                )
        return device_tokens, device_row_offsets

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
        device0 = self.runtime.devices[0]
        pixels = _pinned_copy(blocks, device0, dtype=DType.int32)
        embeddings = self._encode_images(pixels)
        return embeddings, _pinned_copy([indices], device0)

    def _encode_images(self, patches: Buffer) -> Buffer:
        """Runs the vision tower over one block of patches."""
        assert self._vision_model is not None
        embeddings = self._vision_model.execute(patches)[0]
        assert isinstance(embeddings, Buffer)
        return embeddings


def _pinned_copy(
    arrays: Sequence[npt.NDArray[np.generic]],
    device: Device,
    *,
    dtype: DType | None = None,
) -> Buffer:
    """Stacks arrays along their first axis into a copy on device.

    The stack is written straight into pinned host memory, so the copy to the
    device is asynchronous. The host buffer is fresh each call, so a later
    step never overwrites one an earlier copy is still reading.
    """
    host = Buffer(
        shape=(sum(len(array) for array in arrays), *arrays[0].shape[1:]),
        dtype=dtype or DType.from_numpy(arrays[0].dtype),
        device=device,
        usage=Usage.STAGING | Usage.UNTRACKED,
    )
    np.concatenate(arrays, out=host.to_numpy())
    return host.to(device)
