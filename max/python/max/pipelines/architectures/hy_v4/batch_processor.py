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
"""Input batching for the Hy4 and Hy4 VL pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputs, MultiKVCacheParams
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
)
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    ModuleV3SingleReplicaBatchProcessor,
)
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor.compilation import CompiledCallable

from .inputs import HYV4Inputs
from .layers.vision import vision_index_inputs
from .model_config import HYV4Config, HYV4VLConfig
from .mrope import mrope_positions


def prefill_buffer_length(
    contexts: Sequence[TextContext], page_size: int
) -> int:
    """Returns the MLA prefill plan's buffer length for a batch.

    It is ``buffer_lengths[0]``: each request's cache plus new tokens,
    ``current_position``, rounded up to a page and summed.
    """
    return sum(
        (ctx.tokens.current_position + page_size - 1) // page_size * page_size
        for ctx in contexts
    )


class HYV4BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, HYV4Inputs]
):
    """Ragged single-replica batching, plus the host prefill buffer length.

    The length is a host copy, so the graph never reads the plan's back, a host
    sync that device graph capture rejects.
    """

    def __init__(
        self, config: HYV4Config, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        assert isinstance(config.kv_params, MultiKVCacheParams)
        self._page_size = config.kv_params.page_size
        self._batch_context_length = Buffer.zeros(shape=[1], dtype=DType.int32)

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns the inputs, with the host length after the row offsets.

        That is the order ``HYV4.forward`` takes them in.
        """
        inputs = super().get_symbolic_inputs(
            kv_params=kv_params, device_refs=device_refs
        )
        inputs.insert(3, TensorType(DType.int32, [1], device=DeviceRef.CPU()))
        return inputs

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> HYV4Inputs:
        """Builds the first step's inputs from the replica batches."""
        self._batch_context_length.to_numpy()[0] = prefill_buffer_length(
            [ctx for batch in replica_batches for ctx in batch],
            self._page_size,
        )
        return super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> HYV4Inputs:
        return HYV4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            batch_context_length=self._batch_context_length,
        )


def image_grids(ctx: TextContext) -> npt.NDArray[np.int64]:
    """Returns each image's ``(rows, columns)`` patch grid, ``[images, 2]``.

    The image tokenizer stores them in ``extra_model_args``.
    """
    if isinstance(ctx, TextAndVisionContext) and ctx.images:
        return ctx.extra_model_args["image_grids"]
    return np.empty((0, 2), np.int64)


def context_positions(ctx: TextContext, merge: int) -> npt.NDArray[np.int64]:
    """Returns one request's ``[3, active_length]`` M-RoPE positions.

    Up to its last image, a request takes its slice of the whole prompt's
    positions. Past it, the positions are the cache positions plus the
    offset the images add, on all three axes.
    """
    start = ctx.tokens.processed_length
    end = start + ctx.tokens.active_length
    grids = image_grids(ctx)
    if (
        isinstance(ctx, TextAndVisionContext)
        and ctx.images
        and start < ctx.images[-1].end_idx
    ):
        positions, _ = mrope_positions(
            ctx.tokens.all,
            [(1, h, w) for h, w in grids],
            image_token_id=ctx.vision_token_ids[0],
            spatial_merge_size=merge,
        )
        return positions[:, start:end]
    rows, cols = grids[:, 0] // merge, grids[:, 1] // merge + 1
    offset = int(np.sum(np.maximum(rows, cols) - rows * cols))
    return np.tile(np.arange(start, end, dtype=np.int64) + offset, (3, 1))


class HYV4VLBatchProcessor(DecoderBatchProcessor):
    """Batches Hy4 VL with M-RoPE positions and encodes its images.

    The pipeline's vision cache picks each batch's new images. They are
    packed into one tower launch on the first device, and the rows are
    copied to every device.
    """

    _vision_model: CompiledCallable[Any, Any] | None = None
    _empty_vision_embeddings: list[Buffer] | None = None

    @property
    def _vl_config(self) -> HYV4VLConfig:
        assert isinstance(self.config, HYV4VLConfig)
        return self.config

    def token_positions(
        self, context_batch: Sequence[TextContext], position_type: TensorType
    ) -> npt.NDArray[np.integer[Any]]:
        """Returns this step's ``[3, total_seq_len]`` M-RoPE positions."""
        del position_type
        merge = self._vl_config.vision.spatial_merge_size
        return np.concatenate(
            [np.empty((3, 0), np.int64)]
            + [context_positions(ctx, merge) for ctx in context_batch],
            axis=1,
        )

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledCallable[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Binds the compiled tower."""
        del session
        self._vision_model = vision_model

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> list[Buffer] | None:
        """Packs the selected images into the tower's inputs on device 0.

        Returns ``None`` when no image is selected.
        """
        pixels, grids = [], []
        for ctx, miss in selection:
            wanted = {image.start_idx for image in miss}
            for image, grid in zip(ctx.images, image_grids(ctx), strict=True):
                if image.start_idx in wanted:
                    pixels.append(image.pixel_values)
                    grids.append((int(grid[0]), int(grid[1])))
        if not grids:
            return None
        arrays = vision_index_inputs(grids, self._vl_config.vision)
        arrays["pixel_values"] = np.concatenate(pixels)
        max_seqlen = Buffer.from_numpy(arrays.pop("max_seqlen"))
        on = {
            name: Buffer.from_numpy(array).to(devices[0])
            for name, array in arrays.items()
        }
        # The tower's argument order.
        return [
            on["pixel_values"],
            on["pos_idx"],
            on["pos_w"],
            on["input_row_offsets"],
            max_seqlen,
            on["fold_idx"],
            on["newline"],
        ]

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: list[Buffer] | None,
    ) -> VisionEncodeResult:
        """Runs the tower and copies its rows to every device."""
        del selection
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self._vision_model is not None
        (rows,) = self._vision_model.execute_raw(*packed)
        assert isinstance(rows, Buffer)
        return VisionEncodeResult(
            embeddings=[rows.to(device) for device in devices]
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device ``[0, hidden]`` rows for a step that encodes nothing.

        Cached: a captured device graph replays only the same buffers.
        """
        if self._empty_vision_embeddings is None:
            hidden = self._vl_config.vision.out_hidden_size
            self._empty_vision_embeddings = [
                Buffer.zeros(shape=[0, hidden], dtype=DType.bfloat16).to(d)
                for d in devices
            ]
        return self._empty_vision_embeddings
