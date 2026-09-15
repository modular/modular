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

"""Runs a clip through the whole Qwen3VL vision encoder graph on a GPU.

Qwen3.5's vision graph binds ``pixel_values`` and ``rot_pos_ids`` to one
symbolic dimension and the bilinear ``weights``/``indices`` to another, because
the position grid is interpolated once per FRAME while the pixels carry one row
per temporal PATCH. Every ``t == 1`` request makes the two lengths coincide, so
this file is the only place that exercises the split at all.

The batched case is the one that matters. Tiling a clip's position arrays up to
the pixel length -- the workaround a single shared dimension forces -- makes a
lone clip come out right, because ``mo.spatial_merge`` reads only the first
``h * w`` rows of each item. Any grid FOLLOWING that clip is then read from the
wrong input offset and silently mixes in another item's position embeddings. So
the tests below always put a second item after the clip.

The encoder is built at a reduced depth with random weights: the questions here
are shape and offset arithmetic through ``mo.spatial_merge``, the per-temporal-
patch attention segments and the patch merger, none of which depend on trained
values.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.comm import Signals
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen3vl_moe.model_config import VisionConfig
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_bilinear_interpolation_weights_and_indices,
    get_seqlens,
)
from max.pipelines.architectures.qwen3vl_moe.nn.visual_transformer import (
    VisionTransformer,
)

# Small but structurally faithful: merge_size and temporal_patch_size are the
# checkpoint's, and hidden_size stays divisible by the head count.
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
HIDDEN_SIZE = 64
NUM_HEADS = 2
DEPTH = 2
INTERMEDIATE_SIZE = 128
OUT_HIDDEN_SIZE = 32
NUM_POSITION_EMBEDDINGS = 2304
PATCH_ROW_WIDTH = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE**2


def _config(
    device: DeviceRef, deepstack_visual_indexes: list[int]
) -> VisionConfig:
    return VisionConfig(
        dtype=DType.float32,
        llm_dtype=DType.float32,
        devices=[device],
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        in_channels=3,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        depth=DEPTH,
        intermediate_size=INTERMEDIATE_SIZE,
        out_hidden_size=OUT_HIDDEN_SIZE,
        deepstack_visual_indexes=deepstack_visual_indexes,
        rms_norm_eps=1e-6,
        spatial_merge_size=MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
    )


def _random_weights(
    encoder: VisionTransformer, seed: int
) -> dict[str, npt.NDArray[np.float32]]:
    """Small random values for every declared weight, in float32."""
    rng = np.random.default_rng(seed)
    return {
        name: rng.normal(0.0, 0.02, tuple(int(d) for d in weight.shape)).astype(
            np.float32
        )
        for name, weight in encoder.raw_state_dict().items()
    }


@pytest.fixture(
    scope="module",
    params=[
        pytest.param([], id="no_deepstack"),
        pytest.param([0], id="deepstack"),
    ],
)
def compiled(request: pytest.FixtureRequest):  # noqa: ANN201
    """Compiles the vision encoder once, with Qwen3.5's own input bindings.

    The two symbolic dimensions here are the change under test: `vision_pos_len`
    for the position arrays and `vision_seq_len` for the patch rows. Binding
    both to one name is what rejects a clip outright.

    Parameterized on deepstack because the two archs that share this encoder
    differ there: Qwen3.5 declares no deepstack layers, Qwen3-VL-MoE does, and
    a deepstack merger emits its own copy of every merged row.
    """
    device = DeviceRef.GPU(0)
    accelerator = Accelerator(0)
    session = InferenceSession(devices=[accelerator])
    encoder = VisionTransformer(_config(device, request.param))
    # `load_state_dict` is what gives every weight its fully-qualified name;
    # without it several submodules present an unprefixed `weight` and the
    # graph rejects the second one.
    encoder.load_state_dict(
        _random_weights(encoder, seed=11), weight_alignment=1
    )
    signals = Signals(devices=[device])

    with Graph(
        "qwen3_5_vision_clip",
        input_types=(
            TensorType(
                DType.float32,
                shape=["vision_seq_len", PATCH_ROW_WIDTH],
                device=device,
            ),
            TensorType(
                DType.float32, shape=[4, "vision_pos_len", 1], device=device
            ),
            TensorType(DType.int64, shape=[4, "vision_pos_len"], device=device),
            TensorType(DType.int32, shape=["vision_seq_len", 2], device=device),
            TensorType(DType.int32, shape=[], device=DeviceRef.CPU()),
            TensorType(DType.int64, shape=["n_images", 3], device=device),
            TensorType(DType.uint32, shape=["n_seqlens"], device=device),
            TensorType(DType.uint32, shape=[1], device=DeviceRef.CPU()),
            *signals.input_types(),
        ),
    ) as graph:
        (
            pixel_values,
            weights_in,
            indices_in,
            rot_pos_ids,
            max_grid_size,
            grid_thw,
            cu_seqlens,
            max_seqlen,
            *rest,
        ) = graph.inputs
        embeddings, deepstack = encoder(
            pixel_values=[pixel_values.tensor],
            idxs=[indices_in.tensor],
            weights=[weights_in.tensor],
            grid_thw=[grid_thw.tensor],
            rot_pos_ids=[rot_pos_ids.tensor],
            max_grid_size=[max_grid_size.tensor],
            cu_seqlens=[cu_seqlens.tensor],
            max_seqlen=[max_seqlen.tensor],
            signal_buffers=[buf.buffer for buf in rest],
        )
        graph.output(*embeddings, *(t for layer in deepstack for t in layer))

    model = session.load(graph, weights_registry=encoder.state_dict())
    return model, accelerator, signals, len(request.param)


def _pixels(rows: int, seed: int) -> npt.NDArray[np.float32]:
    return (
        np.random.default_rng(seed)
        .normal(0.0, 1.0, (rows, PATCH_ROW_WIDTH))
        .astype(np.float32)
    )


def _encode(
    compiled,  # noqa: ANN001
    grid_thw: npt.NDArray[np.int64],
    pixel_values: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Runs one batch of grid items through the encoder.

    Builds exactly the arrays ``vision_packing`` would: pixels and rotary ids
    at ``sum(t * h * w)`` rows, bilinear weights and indices at ``sum(h * w)``.
    """
    model, accelerator, signals, num_deepstack = compiled

    patch_rows = int(sum(t * h * w for t, h, w in grid_thw))
    assert pixel_values.shape == (patch_rows, PATCH_ROW_WIDTH)
    raw_indices, raw_weights = get_bilinear_interpolation_weights_and_indices(
        grid_thw=grid_thw,
        num_grid_per_side=int(NUM_POSITION_EMBEDDINGS**0.5),
    )
    # The whole point of the two symbolic dimensions: for a clip these arrays
    # are strictly shorter than the pixels.
    plane_rows = int(sum(h * w for _, h, w in grid_thw))
    assert raw_indices.shape == (4, plane_rows)

    grid32 = grid_thw.astype(np.int32)
    cu_seqlens, max_seqlen = get_seqlens(grid32)

    outputs = model.execute(
        Buffer.from_numpy(pixel_values).to(accelerator),
        Buffer.from_numpy(raw_weights.astype(np.float32)).to(accelerator),
        Buffer.from_numpy(raw_indices.astype(np.int64)).to(accelerator),
        Buffer.from_numpy(
            mrope_pos_ids_3d(
                grid_thw=grid32, spatial_merge_size=MERGE_SIZE
            ).astype(np.int32)
        ).to(accelerator),
        Buffer.from_numpy(
            np.array(int(np.max(grid_thw[:, 1:])), dtype=np.int32)
        ),
        Buffer.from_numpy(grid_thw.astype(np.int64)).to(accelerator),
        Buffer.from_numpy(cu_seqlens.astype(np.uint32)).to(accelerator),
        Buffer.from_numpy(np.array([max_seqlen], dtype=np.uint32)),
        *signals.buffers(),
    )
    output = outputs[0]
    assert isinstance(output, Buffer)
    embeddings = output.to(CPU()).to_numpy().astype(np.float32)
    # A deepstack merger emits one row per merged patch, exactly as the final
    # merger does, so a clip's temporal extent has to reach those too.
    assert len(outputs) == 1 + num_deepstack
    for extra in outputs[1:]:
        assert isinstance(extra, Buffer)
        assert tuple(int(d) for d in extra.shape) == embeddings.shape
    return embeddings


def test_single_clip_runs(compiled) -> None:  # noqa: ANN001
    """A clip reaches the encoder at all, and emits one row per merged patch."""
    grid_thw = np.array([(3, 4, 6)], dtype=np.int64)
    embeddings = _encode(compiled, grid_thw, _pixels(3 * 4 * 6, seed=1))
    assert embeddings.shape == (3 * 4 * 6 // MERGE_SIZE**2, OUT_HIDDEN_SIZE)
    assert np.isfinite(embeddings).all()


def test_batched_clip_and_image_run(compiled) -> None:  # noqa: ANN001
    """Two vision items in one batch, the first a clip.

    Row count is the sum over items of ``t * h * w / merge^2``, which is what
    the router spliced placeholders for.
    """
    grid_thw = np.array([(3, 4, 6), (1, 6, 4)], dtype=np.int64)
    rows = 3 * 4 * 6 + 1 * 6 * 4
    embeddings = _encode(compiled, grid_thw, _pixels(rows, seed=2))
    assert embeddings.shape == (rows // MERGE_SIZE**2, OUT_HIDDEN_SIZE)
    assert np.isfinite(embeddings).all()


def test_item_after_a_clip_is_not_corrupted(compiled) -> None:  # noqa: ANN001
    """An image's embeddings do not depend on a clip sitting before it.

    The concrete failure a shared symbolic dimension forces: with the position
    arrays tiled up to the pixel length, ``mo.spatial_merge`` advances its
    input cursor past the tiled copies and reads the trailing image's positions
    from the wrong offset -- mixing in the clip's. Nothing in a single-clip
    batch can see that, because the kernel reads only the first ``h * w`` rows
    of the leading item.

    The image's own pixel rows are byte-identical in both runs, and its
    attention segment is bounded by ``cu_seqlens``, so its embeddings must be
    too.
    """
    clip_grid, image_grid = (3, 4, 6), (1, 6, 4)
    image_pixels = _pixels(image_grid[1] * image_grid[2], seed=3)
    clip_pixels = _pixels(clip_grid[0] * clip_grid[1] * clip_grid[2], seed=4)

    alone = _encode(
        compiled, np.array([image_grid], dtype=np.int64), image_pixels
    )
    batched = _encode(
        compiled,
        np.array([clip_grid, image_grid], dtype=np.int64),
        np.vstack([clip_pixels, image_pixels]),
    )

    trailing = batched[-alone.shape[0] :]
    np.testing.assert_allclose(trailing, alone, rtol=1e-5, atol=1e-5)


def test_clip_position_rows_are_replicated_across_slices(
    compiled,  # noqa: ANN001
) -> None:
    """Identical pixel rows across a clip's slices give identical embeddings.

    The position grid is replicated across temporal patches, and attention is
    segmented per temporal patch, so a clip whose slices hold the same pixels
    must produce the same embeddings in every slice. Had
    ``mo.spatial_merge`` read the position rows at a shifted offset per slice,
    they would diverge.
    """
    grid_t, grid_h, grid_w = 3, 4, 6
    slice_rows = _pixels(grid_h * grid_w, seed=5)
    embeddings = _encode(
        compiled,
        np.array([(grid_t, grid_h, grid_w)], dtype=np.int64),
        np.tile(slice_rows, (grid_t, 1)),
    )

    rows_per_slice = grid_h * grid_w // MERGE_SIZE**2
    assert embeddings.shape == (grid_t * rows_per_slice, OUT_HIDDEN_SIZE)
    slices = embeddings.reshape(grid_t, rows_per_slice, OUT_HIDDEN_SIZE)
    for t in range(1, grid_t):
        np.testing.assert_allclose(slices[t], slices[0], rtol=1e-5, atol=1e-5)
