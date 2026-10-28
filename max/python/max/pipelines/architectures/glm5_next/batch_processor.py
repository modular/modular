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
"""Input batching for GLM-5.3-Flash."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import Buffer, Device, Usage
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessorBase,
)
from max.pipelines.architectures.deepseekV3.model import DeepseekV3Inputs
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.vision_batching import create_empty_image_embeddings
from max.pipelines.lib.vision_encoder_cache import (
    VisionEncodeResult,
    concat_device_buffers,
)

from .context import Glm5NextTextAndVisionContext
from .model_config import Glm5NextConfig
from .nn.vision_geometry import vision_cu_seqlens, vision_position_ids

__all__ = ["Glm5NextBatchProcessor", "Glm5NextInputs"]


class Glm5NextInputs(DeepseekV3Inputs):
    """DeepSeek-V3 inputs with the vision-merge buffers after ``tokens``.

    ``vision_embeddings`` and ``vision_scatter_indices`` are set by the
    pipeline's vision seam on every prepared batch, empties included, so the
    tuple always matches :meth:`.glm5_next.Glm5Next.input_types`.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )


class Glm5NextBatchProcessor(DeepseekV3BatchProcessorBase[Glm5NextInputs]):
    """DeepSeek-V3 batching that builds :class:`Glm5NextInputs`.

    Also encodes images through
    :class:`~max.pipelines.lib.vision_encoder_cache.SupportsVisionEncoding`
    once the compiled tower is bound, so a caller that builds the language
    model without :class:`.model.Glm5NextModel`, as spec decoding does, can
    still run it.
    """

    _vision_model: Model | None = None
    _empty_vision_embeddings: list[Buffer] | None = None

    @property
    def _glm_config(self) -> Glm5NextConfig:
        assert isinstance(self.config, Glm5NextConfig)
        return self.config

    def bind_vision_encoder(
        self, *, vision_model: Model, session: InferenceSession
    ) -> None:
        """Binds the compiled vision tower."""
        del session
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Glm5NextTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> None:
        """Packs inline in :meth:`vision_execute` instead."""
        return

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Glm5NextTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: None,
    ) -> VisionEncodeResult:
        """Encodes the cache-miss images, context-major then image-minor.

        Runs in chunks of at most ``max_batch_input_tokens`` output rows, the
        same bound the decoder honors, so the tower's activations stay within
        what the memory planner reserved. An image is never split.
        """
        assert self._vision_model is not None, (
            "This checkpoint has no vision tower, so it cannot encode images."
        )
        vision_config = self._glm_config.vision_config
        assert vision_config is not None
        merge_unit = vision_config.spatial_merge_size**2

        entries: list[
            tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]
        ] = []
        for ctx, miss_images in selection:
            # Spans never overlap, so a start index names one image even when
            # the miss list holds a reconstructed `ImageMetadata`.
            index_of = {img.start_idx: i for i, img in enumerate(ctx.images)}
            for img in miss_images:
                i = index_of[img.start_idx]
                entries.append((ctx.images[i].pixel_values, ctx.grid_thws[i]))
        counts = [int(np.prod(grid)) // merge_unit for _, grid in entries]

        budget = self.runtime.max_batch_input_tokens
        chunks: list[list[int]] = [[]]
        chunk_rows = 0
        for index, count in enumerate(counts):
            if chunks[-1] and chunk_rows + count > budget:
                chunks.append([])
                chunk_rows = 0
            chunks[-1].append(index)
            chunk_rows += count

        outputs = [
            self._encode_images([entries[i] for i in chunk]) for chunk in chunks
        ]
        embeddings = (
            outputs[0]
            if len(outputs) == 1
            else [
                concat_device_buffers([output[d] for output in outputs])
                for d in range(len(devices))
            ]
        )
        return VisionEncodeResult(
            embeddings=embeddings, per_image_token_counts=counts
        )

    def _encode_images(
        self,
        entries: Sequence[
            tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]
        ],
    ) -> list[Buffer]:
        """Runs the tower once; returns one ``[rows, hidden]`` copy per device."""
        assert self._vision_model is not None
        vision_config = self._glm_config.vision_config
        assert vision_config is not None
        devices = self.runtime.devices
        device0 = devices[0]
        grids = np.stack([grid for _, grid in entries])
        row_offsets = vision_cu_seqlens(grids)
        max_seqlen = np.array(
            [np.diff(row_offsets.astype(np.int64)).max()], dtype=np.uint32
        )
        # Packed straight into staging memory: a concatenated numpy copy is a
        # fresh allocation of up to hundreds of MB per chunk, and on a host
        # with a fragmented page cache its huge-page faults compact
        # synchronously.
        pixels = Buffer(
            dtype=DType.float32,
            shape=(
                sum(len(entry) for entry, _ in entries),
                entries[0][0].shape[1],
            ),
            device=device0,
            usage=Usage.DEFAULT
            if device0.is_host
            else Usage.STAGING | Usage.UNTRACKED,
        )
        staged = pixels.to_numpy()
        row = 0
        for entry, _ in entries:
            staged[row : row + len(entry)] = entry
            row += len(entry)
        outputs = self._vision_model.execute(
            pixels.to(device0),
            Buffer.from_numpy(
                vision_position_ids(grids, vision_config.spatial_merge_size)
            ).to(device0),
            Buffer.from_numpy(row_offsets).to(device0),
            Buffer.from_numpy(max_seqlen),
        )
        assert len(outputs) == len(devices)
        buffers: list[Buffer] = []
        for output in outputs:
            assert isinstance(output, Buffer)
            buffers.append(output)
        return buffers

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device ``[0, hidden]`` rows for a step that encodes nothing.

        Cached, because every decode step asks for them.
        """
        if self._empty_vision_embeddings is None:
            self._empty_vision_embeddings = create_empty_image_embeddings(
                devices, self._glm_config.hidden_size
            )
        return self._empty_vision_embeddings

    def _make_mla_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        host_input_row_offsets: Buffer,
        batch_context_lengths: list[Buffer],
        signal_buffers: list[Buffer],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        return_n_logits: Buffer,
        data_parallel_splits: Buffer,
        ep_inputs: tuple[Buffer, ...],
    ) -> Glm5NextInputs:
        return Glm5NextInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=batch_context_lengths,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
        )
