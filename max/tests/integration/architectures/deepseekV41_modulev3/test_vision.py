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
"""The image path's own checks: weights, inputs, masks and spans.

Every test here is CPU-only and compiles nothing: the vision encoder is
built lazily to read its declared parameters, and the host-side inputs,
launches, masks and spans are checked as data.
"""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import NonCallableMock

import abi_contract
import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.outputs import ReturnLogits
from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    IMAGE_PLACEHOLDER,
)
from max.pipelines.architectures.deepseekV41.image_processor import (
    DeepseekV41ImageProcessor,
)
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
    image_token_mask,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    extra_input_names,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41_modulev3.tokenizer import (
    DeepseekV41Tokenizer,
)
from max.pipelines.architectures.deepseekV41_modulev3.vision_encoder import (
    DeepseekV41VisionConfig,
    DeepseekV41VisionModel,
    vision_input_types,
)
from max.pipelines.architectures.deepseekV41_modulev3.vision_inputs import (
    IMAGE_GRIDS,
    launch_bounds,
    pack_uncached_images,
    span_tokens,
    vision_inputs,
)
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.tensor import default_dtype
from vision_fixtures import (
    IMAGE,
    byte_tokenizer,
    chat_request,
    png,
    vision_config,
)


def test_encoder_declares_the_checkpoints_vision_tensors() -> None:
    """Mach feeds the encoder the checkpoint's own tensors, byte for byte.

    Every parameter must be a census tensor of the same dtype and shape,
    and every vision census tensor must have a parameter.
    """
    with F.lazy(), default_dtype(DType.bfloat16):
        encoder = DeepseekV41VisionModel(vision_config())
    declared = {
        name: (tensor.dtype, tuple(int(d) for d in tensor.shape))
        for name, tensor in encoder.parameters
    }
    census = abi_contract.census(
        V41Config.from_hf(abi_contract.hf_config("full"))
    )
    vision = {
        name: entry
        for name, entry in census.items()
        if name.startswith(("vision.", "aligner.", "image_"))
    }
    assert declared == vision


def test_encoder_input_types_name_the_packed_dims() -> None:
    config = vision_config()
    types = vision_input_types(config, DeviceRef.GPU(0), DType.bfloat16)
    assert [t.dtype for t in types] == [
        DType.bfloat16,
        DType.float32,
        DType.float32,
        DType.uint32,
        DType.uint32,
        DType.int64,
        DType.int64,
    ]
    assert [str(d) for d in types[0].shape] == ["total_patches", "588"]
    assert types[4].device == DeviceRef.CPU()


def test_packed_inputs_cover_each_images_patches_and_span() -> None:
    """Two images: a 4x5 patch grid that pads to 2x2 blocks, and a 3x3."""
    config = vision_config()
    grids = [(4, 5), (3, 3)]
    host = vision_inputs(grids, config)
    patches = 4 * 5 + 3 * 3
    assert host.rot_cos.shape == (patches, config.rope_dim)
    assert host.rot_sin.dtype == np.float32
    assert host.cu_seqlens.tolist() == [0, 20, 29]
    assert int(host.max_seqlen) == 20
    # One aligner row per 3x3 block: 2x2 blocks, then 1x1.
    assert host.aligner_index.shape == ((4 + 1) * 9,)
    # Padded grid cells read the zero row past the last patch.
    assert int(host.aligner_index.max()) == patches
    spans = [span_tokens(h, w, 3) for h, w in grids]
    assert spans == [2 * 3 + 2, 1 * 2 + 2]
    assert host.span_index.shape == (sum(spans),)
    # Rows 0-4 are aligner rows; 5, 6 and 7 the start, newline and end.
    assert host.span_index.tolist() == [5, 0, 1, 6, 2, 3, 6, 7, 5, 4, 6, 7]


def test_vision_rope_theta_may_be_a_float() -> None:
    full = abi_contract.hf_config("full")
    vision = {**full["vision_config"], "rope_theta": 10000.0}
    hidden_size = full["text_config"]["hidden_size"]
    config = DeepseekV41VisionConfig.from_hf(vision, hidden_size)
    assert config.rope_theta == 10000.0
    with pytest.raises(ValueError, match="rope_theta"):
        DeepseekV41VisionConfig.from_hf(
            {**vision, "rope_theta": "10000"}, hidden_size
        )
    with pytest.raises(ValueError, match="patch_size"):
        DeepseekV41VisionConfig.from_hf(
            {**vision, "patch_size": 14.0}, hidden_size
        )


def test_a_launch_takes_whole_images_up_to_the_budget() -> None:
    assert launch_bounds([(3, 3), (3, 3), (2, 2), (3, 3)], 18) == [
        (0, 2),
        (2, 4),
    ]
    assert launch_bounds([(3, 3)] * 3, 9) == [(0, 1), (1, 2), (2, 3)]
    assert launch_bounds([], 9) == []
    with pytest.raises(ValueError, match="more than"):
        launch_bounds([(3, 4)], 9)


def _image_context(
    grids: list[tuple[int, int]], fills: list[int]
) -> TextAndVisionContext:
    """Back-to-back image spans; ``fills`` gives each image's patch value."""
    images, start = [], 0
    for (h, w), fill in zip(grids, fills, strict=True):
        end = start + span_tokens(h, w, 3)
        images.append(
            ImageMetadata(
                start_idx=start,
                end_idx=end,
                pixel_values=np.full((h * w, 3, 14, 14), fill, np.uint16),
            )
        )
        start = end
    return TextAndVisionContext(
        max_length=64,
        tokens=TokenBuffer(np.full(start, IMAGE, np.int64)),
        images=images,
        vision_token_ids=[IMAGE],
        extra_model_args={IMAGE_GRIDS: np.array(grids, np.int64)},
    )


def test_each_launch_packs_its_own_images() -> None:
    """At most one largest image's patches per launch: 9 + 9, then 18."""
    config = dataclasses.replace(vision_config(), max_image_tokens=2)
    assert config.max_image_patches == 18
    first = _image_context([(3, 3), (3, 3)], [1, 2])
    second = _image_context([(3, 6)], [3])
    launches = pack_uncached_images(
        [(first, first.images), (second, second.images)], CPU(), config
    )
    assert launches is not None
    assert len(launches) == 2
    for launch, grids in zip(
        launches, [[(3, 3), (3, 3)], [(3, 6)]], strict=True
    ):
        host = vision_inputs(grids, config)
        assert launch[0].shape == (sum(h * w for h, w in grids), 588)
        np.testing.assert_array_equal(launch[3].to_numpy(), host.cu_seqlens)
        np.testing.assert_array_equal(launch[5].to_numpy(), host.aligner_index)
        np.testing.assert_array_equal(launch[6].to_numpy(), host.span_index)
    for launch, fills in zip(
        launches, [[1] * 9 + [2] * 9, [3] * 18], strict=True
    ):
        patches = launch[0].view(DType.uint16).to_numpy()
        np.testing.assert_array_equal(patches[:, 0], fills)


def _context(tokens: list[int], processed: int = 0) -> TextContext:
    ctx = TextContext(
        max_length=64, tokens=TokenBuffer(np.array(tokens, dtype=np.int64))
    )
    ctx.tokens.skip_processing(processed)
    return ctx


def test_mask_marks_the_image_rows_of_each_request() -> None:
    batch = [_context([1, IMAGE, IMAGE, 2]), _context([IMAGE, 3], 1)]
    assert image_token_mask(batch, IMAGE).tolist() == [0, 1, 1, 0, 0]
    # A verify's draft rows follow each request's own rows, as text.
    assert image_token_mask(batch, IMAGE, verify_width=2).tolist() == [
        *[0, 1, 1, 0, 0, 0],
        *[0, 0, 0],
    ]
    assert image_token_mask(batch, None).tolist() == [0] * 5


def _processor(config: object) -> DeepseekV41BatchProcessor:
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(
            Any,
            SimpleNamespace(
                model=SimpleNamespace(
                    huggingface_config=abi_contract.hf_config("f1"),
                    data_parallel_degree=1,
                )
            ),
        ),
        devices=[CPU()],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=8,
        max_seq_len=64,
        max_batch_input_tokens=64,
    )
    processor = DeepseekV41BatchProcessor(cast(Any, config), runtime)
    names = extra_input_names(V41Config.from_hf(abi_contract.hf_config("f1")))
    static = {
        name: Buffer.zeros(
            [1],
            DType.float8_e4m3fn
            if name.endswith("_weight")
            else DType.float8_e8m0fnu,
        )
        for name in names
        if name.startswith("engram_table_")
    }
    static["engram_token_map"] = Buffer.zeros([1], DType.int32)
    processor.bind_engram(static)
    return processor


def _model_config(**fields: Any) -> DeepseekV41Config:
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=NonCallableMock(),
        devices=[DeviceRef.CPU()],
        max_seq_len=64,
        **fields,
    )


def test_extras_mark_image_rows_and_dead_lookback() -> None:
    """The chunk after an image span: its lookback reads image tokens."""
    processor = _processor(_model_config(image_token_id=IMAGE))
    batch = [
        _context([7, IMAGE, IMAGE, 8, 9], 3),
        _context([IMAGE, IMAGE, 5]),
    ]
    extras = processor._extras(batch, np.array([0, 2, 5]))
    by_name = dict(
        zip(
            extra_input_names(V41Config.from_hf(abi_contract.hf_config("f1"))),
            extras,
            strict=True,
        )
    )
    assert by_name["image_token_mask"].to_numpy().tolist() == [0, 0, 1, 1, 0]
    assert by_name["engram_lookback"].to_numpy().tolist() == [
        [IMAGE, IMAGE, 7],
        [-1, -1, -1],
    ]
    assert by_name["engram_lookback_dead"].to_numpy().tolist() == [
        [1, 1, 0],
        [0, 0, 0],
    ]


def _tokenizer(*, serves_images: bool = True) -> DeepseekV41Tokenizer:
    delegate = byte_tokenizer()
    image_token_id = delegate.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    tokenizer = object.__new__(DeepseekV41Tokenizer)
    tokenizer.delegate = delegate
    tokenizer.max_length = 4096
    tokenizer._token_id_bound = len(delegate)
    tokenizer._eos_token_ids = set()
    tokenizer._chat_template = None
    tokenizer.image_token_id = image_token_id
    tokenizer.vision_token_ids = [image_token_id]
    tokenizer.image_processor = (
        DeepseekV41ImageProcessor(
            SimpleNamespace(
                patch_size=14,
                downsample_ratio=3,
                min_pixels=0,
                max_image_tokens=1024,
                max_wh_ratio=None,
            )
        )
        if serves_images
        else None
    )
    return tokenizer


def test_chat_image_grows_into_its_span() -> None:
    """A 42x84 image is a 3x6 patch grid, a 1x2 aligner grid, 5 positions."""
    tokenizer = _tokenizer()
    context = asyncio.run(
        tokenizer.new_context(chat_request("what is this?", [png(84, 42)]))
    )
    (image,) = context.images
    assert image.end_idx - image.start_idx == span_tokens(3, 6, 3) == 5
    ids = context.tokens.array
    assert (
        ids[image.start_idx : image.end_idx].tolist()
        == [tokenizer.image_token_id] * 5
    )
    assert int(np.sum(ids == tokenizer.image_token_id)) == 5
    assert context.extra_model_args[IMAGE_GRIDS].tolist() == [[3, 6]]
    assert image.pixel_values.shape == (18, 3, 14, 14)


def test_text_only_deployment_refuses_images() -> None:
    tokenizer = _tokenizer(serves_images=False)
    with pytest.raises(InputError, match="serves text only"):
        asyncio.run(
            tokenizer.new_context(chat_request("what is this?", [png(42, 42)]))
        )


def test_token_prompt_with_the_image_token_is_an_input_error() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.image_token_id is not None
    with pytest.raises(InputError, match="image token"):
        asyncio.run(tokenizer.encode([1, tokenizer.image_token_id, 2]))
