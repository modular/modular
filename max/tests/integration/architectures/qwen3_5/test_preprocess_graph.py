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
"""Checks the Qwen3.5 image and video preprocessing graphs against MAX
serve's.

A serving engine that runs the exported graphs must feed the tower what MAX
serve does: the tokenizer's patches, position tables and rotary positions,
and the grid and segment bounds the model packs for them. For video, each
temporal patch of a clip is one run, which the tower attends over alone.
Every output but the patches matches bit for bit. The graphs resize uint8
pixels, accumulating in float32 and rounding once at the end, where the
tokenizer resizes an image, or each frame of a clip, with Pillow, which
rounds and clamps between its two passes, so the patches match within a
level or two, and further around a chart's hard edges. The decoder's M-RoPE
positions the graphs return for each item, stitched into a prompt, are the
ones ``get_rope_index`` gives it.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max import tree
from max.driver import CPU, Buffer
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen3_5.layers.visual_transformer import (
    Qwen3_5VisionTransformer,
)
from max.pipelines.architectures.qwen3_5.model import Qwen3_5Model
from max.pipelines.architectures.qwen3_5.preprocess_graph import (
    Qwen3_5ImagePreprocess,
)
from max.pipelines.architectures.qwen3_5.vision_packing import (
    pack_uncached_images,
)
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    QWEN3VL_MAX_PIXELS,
    get_bilinear_interpolation_weights_and_indices,
    get_rope_index,
)
from max.pipelines.architectures.qwen3vl_moe.nn.video_processing import (
    clip_grid,
    patchify_clip,
)
from max.pipelines.architectures.qwen3vl_moe.tokenizer import (
    Qwen3VLImageProcessor,
)
from max.pipelines.lib.graph_specs import (
    GraphRole,
    GraphSpec,
    compile_and_init_graphs,
)
from max.tensor.compilation import CompiledCallable
from PIL import Image
from test_common.vision_preprocess_parity import (
    chart,
    check,
    gradient,
    graph_inputs,
    noise,
    palette_with_transparency,
    scene,
)

# The Qwen3.5 checkpoints' vision config.
_PATCH = 16
_TEMPORAL_PATCH = 2
# How far one uint8 level moves a patch value, which the tokenizer
# normalizes as (level - 127.5) / 127.5.
_LEVEL = 1 / 127.5
_MERGE = 2
_POSITION_EMBEDDINGS = 2304

# Prompt token ids for get_rope_index, which knows only these three and
# counts an item by the token after each vision start.
_TEXT, _VISION_START, _IMAGE, _VIDEO = 0, 1, 2, 3


@pytest.fixture(scope="module")
def preprocess() -> CompiledCallable[..., Any]:
    spec = GraphSpec(
        Qwen3_5ImagePreprocess(
            _PATCH, _TEMPORAL_PATCH, _MERGE, _POSITION_EMBEDDINGS
        ),
        "qwen3_5_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )
    role = GraphRole.VISION_PREPROCESS
    session = InferenceSession(devices=[CPU()])
    return compile_and_init_graphs(session, {role: spec})[role]


@pytest.fixture(scope="module")
def preprocess_video() -> CompiledCallable[..., Any]:
    model: Any = SimpleNamespace(
        huggingface_config=SimpleNamespace(vision_config=_vision_config())
    )
    spec = Qwen3_5Model.vision_preprocess_video_spec(model)
    assert spec is not None
    role = GraphRole.VISION_PREPROCESS_VIDEO
    session = InferenceSession(devices=[CPU()])
    return compile_and_init_graphs(session, {role: spec})[role]


_CASES: dict[str, Callable[[], Image.Image]] = {
    # The resize is skipped at the target size.
    "identity_512x384": lambda: noise(384, 512, "RGB"),
    # Each side rounds to the nearest multiple of 32.
    "round_501x333": lambda: scene(333, 501),
    "round_down_2400x1615": lambda: gradient(1615, 2400),
    # Upscaled to the minimum pixel count.
    "upscale_47x61": lambda: scene(61, 47),
    "tiny_9x13": lambda: gradient(9, 13),
    "extreme_1000x40": lambda: gradient(40, 1000),
    "rgba": lambda: scene(300, 202, "RGBA"),
    "gray": lambda: scene(200, 320, "L"),
    "gray_alpha": lambda: scene(260, 240, "LA"),
    "palette_transparency": palette_with_transparency,
}

# An image shrinks only past QWEN3VL_MAX_PIXELS, so the chart clip, which
# the clip budget shrinks, covers downscaling.
_CHARTS: dict[str, Callable[[], Image.Image]] = {
    "round_501x333": lambda: chart(333, 501),
    "upscale_47x61": lambda: chart(61, 47),
}


def _processed(
    image: Image.Image,
) -> tuple[tuple[int, int], npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """The image's resized size, grid and patch rows, as MAX serve's
    tokenizer computes them."""
    # The tokenizer's image loader converts every image to RGB.
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    processor = Qwen3VLImageProcessor(
        patch_size=_PATCH,
        temporal_patch_size=_TEMPORAL_PATCH,
        merge_size=_MERGE,
        min_pixels=65536,
        max_pixels=QWEN3VL_MAX_PIXELS,
    )
    _, grid_thw, (pixel_values,) = processor([rgb])
    _, grid_h, grid_w = (int(n) for n in grid_thw[0])
    return (
        (grid_h * _PATCH, grid_w * _PATCH),
        grid_thw.astype(np.int64),
        pixel_values,
    )


def _served(image: Image.Image) -> tuple[tuple[int, int], list[Buffer]]:
    """What MAX serve feeds the tower for the image and the decoder
    positions it gives the image, and the image's resized size."""
    size, grid_thw, pixel_values = _processed(image)
    served = _packed(pixel_values, grid_thw) + _positions(grid_thw, _IMAGE)
    return size, served


def _rope_index(
    prompt: list[int],
    image_grid_thw: npt.NDArray[np.int64] | None = None,
    video_grid_thw: npt.NDArray[np.int64] | None = None,
) -> tuple[npt.NDArray[np.int64], int]:
    """MAX serve's ``[3, len(prompt)]`` M-RoPE positions and rope delta."""
    positions, delta = get_rope_index(
        spatial_merge_size=_MERGE,
        image_token_id=_IMAGE,
        video_token_id=_VIDEO,
        vision_start_token_id=_VISION_START,
        input_ids=np.array([prompt]),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    return positions[:, 0], int(delta.item())


def _positions(grid_thw: npt.NDArray[np.int64], token: int) -> list[Buffer]:
    """The positions MAX serve gives one item's placeholder tokens,
    relative to the position the first one starts at, and how far the item
    moves the text after it: ``get_rope_index`` over the item alone."""
    _, grid_h, grid_w = (int(n) for n in grid_thw[0])
    tokens = grid_h * grid_w // _MERGE**2
    grids = {
        "image_grid_thw" if token == _IMAGE else "video_grid_thw": grid_thw
    }
    positions, _ = _rope_index(
        [_TEXT, _VISION_START, *[token] * tokens, _TEXT], **grids
    )
    start = positions[0, 1] + 1
    return [
        Buffer.from_numpy(np.ascontiguousarray(positions[:, 2:-1] - start)),
        Buffer.from_numpy(np.array(positions[0, -1] - start, np.int64)),
    ]


def _packed(
    pixel_values: npt.NDArray[np.float32],
    grid_thw: npt.NDArray[np.integer[Any]],
) -> list[Buffer]:
    """The tower's inputs MAX serve packs for one item's patch rows."""
    indices, weights = get_bilinear_interpolation_weights_and_indices(
        grid_thw=grid_thw, num_grid_per_side=int(_POSITION_EMBEDDINGS**0.5)
    )
    marker: Any = object()
    context: Any = SimpleNamespace(
        request_id="request",
        images=[marker],
        vision_data=SimpleNamespace(
            image_grid_thw=grid_thw,
            concatenated_pixel_values=pixel_values,
            weights=weights,
            indices=indices,
            vision_position_ids=mrope_pos_ids_3d(
                grid_thw=grid_thw, spatial_merge_size=_MERGE
            ),
        ),
    )
    packed = pack_uncached_images([(context, [marker])], [CPU()])
    assert packed is not None
    return [
        packed.pixel_values,
        packed.weights,
        packed.indices,
        packed.vision_position_ids,
        packed.max_grid_size,
        packed.grid_thw,
        packed.cu_seqlens,
        packed.max_seqlen,
    ]


@pytest.mark.parametrize("case", sorted(_CASES))
def test_matches_the_served_inputs(
    preprocess: CompiledCallable[..., Any], case: str
) -> None:
    image = _CASES[case]()
    size, served = _served(image)
    outputs = preprocess.execute_raw(*graph_inputs(image, size))
    check(preprocess, case, outputs, served, _LEVEL)


@pytest.mark.parametrize("case", sorted(_CHARTS))
def test_a_chart_matches_the_served_inputs(
    preprocess: CompiledCallable[..., Any], case: str
) -> None:
    """A chart's text, lines and bars, whose hard edges near black and white
    Pillow clamps between its passes, match within the hard-edge bounds."""
    image = _CHARTS[case]()
    size, served = _served(image)
    outputs = preprocess.execute_raw(*graph_inputs(image, size))
    check(preprocess, f"chart_{case}", outputs, served, _LEVEL, hard_edges=True)


# Frame count, height and width of each synthetic clip.
_CLIPS: dict[str, tuple[int, int, int]] = {
    # The frames are already the clip's target size, so MAX skips the resize.
    "identity_4f_512x384": (4, 384, 512),
    # An odd count pads the last temporal patch with the last frame.
    "odd_5f_1280x720": (5, 720, 1280),
    "round_6f_501x333": (6, 333, 501),
    # Upscaled to the clip's minimum pixel count.
    "upscale_2f_40x56": (2, 56, 40),
    # So many frames that the clip budget shrinks each one.
    "budget_14f_1920x1080": (14, 1080, 1920),
}


def _clip(
    frames: list[Image.Image],
) -> tuple[tuple[int, int, int], npt.NDArray[np.float32], list[list[Buffer]]]:
    """The clip's grid and patch rows as MAX serve's video processor
    computes them, and the graph inputs a serving engine passes for each
    temporal patch: its frames, the last repeated to fill it, and the clip's
    target size."""
    pixel_values, grid = patchify_clip(frames)
    _, (h_bar, w_bar) = clip_grid(len(frames), *frames[0].size[::-1])
    rgba = np.stack(
        [np.asarray(frame.convert("RGBA")) for frame in frames]
        + [np.asarray(frames[-1].convert("RGBA"))]
        * (len(frames) % _TEMPORAL_PATCH)
    )
    size = Buffer.from_numpy(np.array([[h_bar, w_bar]], np.int64))
    patches = [
        [
            Buffer.from_numpy(
                np.ascontiguousarray(rgba[t : t + _TEMPORAL_PATCH])
            ),
            size,
        ]
        for t in range(0, len(rgba), _TEMPORAL_PATCH)
    ]
    return grid, pixel_values, patches


@pytest.mark.parametrize("case", sorted(_CLIPS))
def test_each_temporal_patch_matches_the_served_clip(
    preprocess_video: CompiledCallable[..., Any], case: str
) -> None:
    """The graph's run over one temporal patch gives the tower that patch's
    rows of MAX serve's clip, and the inputs MAX packs for them as a
    one-patch grid, and the decoder the positions MAX gives its run."""
    count, height, width = _CLIPS[case]
    frames = [scene(height, width, seed=seed) for seed in range(count)]
    _check_clip(preprocess_video, case, frames)


def test_a_chart_clip_matches_the_served_clip(
    preprocess_video: CompiledCallable[..., Any],
) -> None:
    """A clip of charts, downscaled to the clip budget, matches within the
    hard-edge bounds."""
    count, height, width = _CLIPS["budget_14f_1920x1080"]
    frames = [chart(height, width, seed=seed) for seed in range(count)]
    _check_clip(preprocess_video, "chart_budget_14f", frames, hard_edges=True)


def _check_clip(
    preprocess_video: CompiledCallable[..., Any],
    case: str,
    frames: list[Image.Image],
    hard_edges: bool = False,
) -> None:
    """Checks the graph's run over each temporal patch of the clip."""
    (grid_t, grid_h, grid_w), pixel_values, patches = _clip(frames)
    assert len(patches) == grid_t
    rows = grid_h * grid_w
    one_patch = np.array([[1, grid_h, grid_w]], np.int64)
    for t, inputs in enumerate(patches):
        outputs = preprocess_video.execute_raw(*inputs)
        served = _packed(
            pixel_values[t * rows : (t + 1) * rows], one_patch
        ) + _positions(one_patch, _VIDEO)
        check(
            preprocess_video,
            f"{case}[{t}]",
            outputs,
            served,
            _LEVEL,
            hard_edges=hard_edges,
        )


def _stitch(
    length: int, runs: list[tuple[int, list[Buffer]]]
) -> tuple[npt.NDArray[np.int64], int]:
    """Lays a prompt of ``length`` tokens out as a serving engine does from
    each run's start and its graph's outputs: text takes the running
    position on every axis, and a run the running position plus its own,
    then moves it by its advance."""
    positions = np.empty((3, length), np.int64)
    running = cursor = 0
    for start, outputs in runs:
        ids, advance = outputs[-2].to_numpy(), int(outputs[-1].to_numpy())
        positions[:, cursor:start] = running + np.arange(start - cursor)
        running += start - cursor
        positions[:, start : start + ids.shape[1]] = running + ids
        running += advance
        cursor = start + ids.shape[1]
    positions[:, cursor:] = running + np.arange(length - cursor)
    running += length - cursor
    return positions, running - length


def test_stitched_positions_match_get_rope_index(
    preprocess: CompiledCallable[..., Any],
    preprocess_video: CompiledCallable[..., Any],
) -> None:
    """The graphs' positions for the images and the clip's temporal patches
    of a prompt, stitched together, are the prompt's ``get_rope_index``
    positions and rope delta. The clip's patches sit apart, each after a
    timestamp label, as Qwen3.5's prompt lays them out."""
    prompt: list[int] = [_TEXT] * 3
    runs: list[tuple[int, list[Buffer]]] = []

    def splice(outputs: list[Buffer], token: int) -> None:
        prompt.append(_VISION_START)
        runs.append((len(prompt), outputs))
        prompt.extend([token] * outputs[-2].shape[1])
        prompt.append(_TEXT)

    image_grids = []

    def splice_image(image: Image.Image) -> None:
        size, grid_thw, _ = _processed(image)
        image_grids.append(grid_thw)
        splice(preprocess.execute_raw(*graph_inputs(image, size)), _IMAGE)
        prompt.extend([_TEXT] * 2)

    splice_image(noise(333, 501, "RGB"))
    splice_image(gradient(40, 1000))
    grid, _, patches = _clip([noise(120, 90, "RGB", s) for s in range(5)])
    for inputs in patches:
        prompt.extend([_TEXT] * 4)
        splice(preprocess_video.execute_raw(*inputs), _VIDEO)
    splice_image(gradient(9, 13))

    want, want_delta = _rope_index(
        prompt,
        image_grid_thw=np.concatenate(image_grids),
        video_grid_thw=np.array([grid], np.int64),
    )
    got, delta = _stitch(len(prompt), runs)
    np.testing.assert_array_equal(got, want)
    assert delta == want_delta


def _vision_config() -> SimpleNamespace:
    return SimpleNamespace(
        patch_size=_PATCH,
        temporal_patch_size=_TEMPORAL_PATCH,
        spatial_merge_size=_MERGE,
        num_position_embeddings=_POSITION_EMBEDDINGS,
    )


@pytest.mark.parametrize(
    "spec_of",
    [
        Qwen3_5Model.vision_preprocess_spec,
        Qwen3_5Model.vision_preprocess_video_spec,
    ],
)
def test_vision_preprocess_returns_the_vision_inputs_by_path(
    spec_of: Callable[[Any], GraphSpec | None],
) -> None:
    """A serving engine feeds the preprocess graphs' outputs to the tower by
    path, so they are the tower's one-device inputs for an image or a
    temporal patch, all but the TP signal buffers the engine supplies."""
    model: Any = SimpleNamespace(
        huggingface_config=SimpleNamespace(vision_config=_vision_config())
    )
    spec = spec_of(model)
    assert spec is not None
    outputs, out_tree = tree.flatten(spec.trace().output_tree)
    tower: Any = SimpleNamespace(
        devices=[DeviceRef.CPU()],
        patch_embed=SimpleNamespace(
            patch_dim=3 * _TEMPORAL_PATCH * _PATCH * _PATCH
        ),
    )
    inputs, in_tree = tree.flatten(Qwen3_5VisionTransformer.input_tree(tower))
    image_inputs = [
        (path, leaf)
        for path, leaf in zip(in_tree.leaf_paths, inputs, strict=True)
        if not path.startswith("signals.")
    ]
    # The item's decoder positions follow the tower's inputs.
    assert list(out_tree.leaf_paths) == [path for path, _ in image_inputs] + [
        "positions.position_ids",
        "positions.position_advance",
    ]
    for output, (_, want) in zip(outputs, image_inputs, strict=False):
        assert (output.dtype, output.rank) == (want.dtype, want.rank)
        assert output.device == DeviceRef.CPU()


def test_a_checkpoint_without_a_tower_has_no_preprocess_graph() -> None:
    model: Any = SimpleNamespace(huggingface_config=SimpleNamespace())
    assert Qwen3_5Model.vision_preprocess_spec(model) is None
    assert Qwen3_5Model.vision_preprocess_video_spec(model) is None
