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
"""Structural tests for Kimi K2.5's registered vision seam.

Kimi K2.5 registers its decoder and its vision tower by role, so the pipeline
model the framework builds for it implements ``SupportsVisionEncoding`` and
delegates image encoding to ``KimiK2_5BatchProcessor``. The pipeline's
vision seam (``VisionEncoderCache.finalize_vision_inputs``) then sets the
base ``vision_embeddings`` / ``vision_scatter_indices`` fields, which the
decoder's inputs pass right after the tokens. These CPU-only tests pin that
seam without weights or GPUs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import NonCallableMock

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.pipelines.architectures.kimik2_5_modulev3 import (
    KimiK2_5Config,
    kimik2_5_modulev3_arch,
)
from max.pipelines.architectures.kimik2_5_modulev3.batch_processor import (
    KimiK2_5BatchProcessor,
)
from max.pipelines.architectures.kimik2_5_modulev3.layers.vision.transformer import (
    KimiK2_5VisionEncoder,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.module_registration import (
    DecoderModelInputs,
    ModuleV3DecoderPipelineModel,
)
from max.pipelines.lib.vision_encoder_cache import (
    SupportsVisionEncoding,
    VisionEncoderCache,
    vision_encoding,
)

_N_DEVICES = 4
_HIDDEN_SIZE = 64


def _bare_batch_inputs(n_devices: int) -> DecoderModelInputs:
    """Text inputs as the batch processor prepares them, before the vision
    seam sets the image embeddings."""
    return DecoderModelInputs(
        tokens=Buffer.zeros((1,), dtype=DType.int64),
        input_row_offsets=Buffer.zeros((2,), dtype=DType.uint32),
        return_n_logits=Buffer.zeros((1,), dtype=DType.int64),
        kv_cache_inputs=None,
        batch_context_lengths=[
            Buffer.zeros((1,), dtype=DType.int32) for _ in range(n_devices)
        ],
    )


def _fake_pipeline_model(n_devices: int) -> Any:
    """Kimi K2.5's registered pipeline model without ``__init__``, exposing
    only what the prepare and finalize path reads."""
    model_cls = kimik2_5_modulev3_arch.pipeline_model
    assert isinstance(model_cls, type)
    model = cast(Any, object.__new__(model_cls))
    model.devices = [CPU() for _ in range(n_devices)]
    processor = cast(Any, object.__new__(KimiK2_5BatchProcessor))
    processor.config = NonCallableMock(
        spec=KimiK2_5Config,
        llm_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE),
    )
    processor.bind_vision_encoder(vision_model=object(), session=object())
    model._batch_processor = processor
    return model


def test_model_encodes_images_through_its_batch_processor() -> None:
    """The registered pipeline model hands image encoding to its batching.

    Otherwise the pipeline's vision-encoding gate never fires and no image
    is encoded.
    """
    model_cls = kimik2_5_modulev3_arch.pipeline_model
    assert isinstance(model_cls, type)
    assert issubclass(model_cls, ModuleV3DecoderPipelineModel)
    assert model_cls.vision_cls is KimiK2_5VisionEncoder
    model = _fake_pipeline_model(_N_DEVICES)
    encoding = vision_encoding(model)
    assert isinstance(encoding, SupportsVisionEncoding)
    assert encoding is model._batch_processor


def test_finalized_inputs_pass_the_embeddings_after_the_tokens() -> None:
    """Finalized inputs carry the vision seam's embeddings, which
    ``.buffers`` passes in the decoder's canonical order."""
    model = _fake_pipeline_model(_N_DEVICES)
    inputs = _bare_batch_inputs(_N_DEVICES)
    assert inputs.vision_embeddings == []
    assert inputs.vision_scatter_indices == []

    encoder_cache: VisionEncoderCache[Any] = VisionEncoderCache(
        devices=model.devices
    )
    encoding = vision_encoding(model)
    assert encoding is not None
    encoder_cache.finalize_vision_inputs(encoding, inputs, model.devices, None)

    embeddings = inputs.vision_embeddings
    indices = inputs.vision_scatter_indices
    assert len(embeddings) == _N_DEVICES
    assert len(indices) == _N_DEVICES
    for embed in embeddings:
        assert tuple(embed.shape) == (0, _HIDDEN_SIZE)
    for index in indices:
        assert tuple(index.shape) == (0,)
        assert index.dtype == DType.int32

    # tokens, vision, input_row_offsets, return_n_logits, kv (none here),
    # batch_context_lengths.
    assert inputs.buffers == (
        inputs.tokens,
        *embeddings,
        *indices,
        inputs.input_row_offsets,
        inputs.return_n_logits,
        *(inputs.batch_context_lengths or ()),
    )


def test_collect_uncached_image_inputs_matches_by_span() -> None:
    """Miss images are collected by span position, not object identity.

    Mirrors the non-ModuleV3 sibling's MXSERV-330 regression test: a
    selection whose miss list holds reconstructed ``ImageMetadata`` instances
    for the same logical image must still collect a per-image entry.
    """
    processor = cast(Any, object.__new__(KimiK2_5BatchProcessor))
    pixels = np.zeros((4, 8), dtype=np.float32)
    stored = ImageMetadata(
        start_idx=2, end_idx=4, pixel_values=pixels, image_hash=0xB
    )
    reconstructed = ImageMetadata(
        start_idx=2, end_idx=4, pixel_values=pixels, image_hash=0xB
    )
    ctx = SimpleNamespace(
        images=[stored],
        grid_thws=[(1, 2, 2)],
        position_ids=np.arange(4, dtype=np.int64),
    )

    per_image = processor._collect_uncached_image_inputs(
        [(cast(Any, ctx), [reconstructed])]
    )

    assert len(per_image) == 1
    assert per_image[0]["pixel_values"] is pixels
    np.testing.assert_array_equal(per_image[0]["grid_thw"], [1, 2, 2])
    np.testing.assert_array_equal(per_image[0]["position_ids"], [0, 1, 2, 3])
