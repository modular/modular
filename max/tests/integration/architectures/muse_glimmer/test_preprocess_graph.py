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
"""Checks the Muse Glimmer image preprocessing graph against MAX serve's.

A serving engine that runs the exported graph must feed the tower what MAX
serve does: the image processor's patches, and the position taps, window
order, rotary tables and segment bounds the batch processor packs for them.
Every output matches bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.architectures.muse_glimmer.batch_vision_inputs import (
    IMAGE_GRID_THW,
    pack_uncached_images,
)
from max.pipelines.architectures.muse_glimmer.image_processor import (
    MuseGlimmerImageProcessor,
)
from max.pipelines.architectures.muse_glimmer.model_config import (
    MuseGlimmerVisionConfig,
)
from max.pipelines.architectures.muse_glimmer.preprocess_graph import (
    MuseGlimmerImagePreprocess,
)
from max.pipelines.architectures.muse_glimmer.vision import vision_input_tree
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

# The checkpoint's processor_config.json image_processor block.
_PROCESSOR = MuseGlimmerImageProcessor(
    {
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "max_image_tokens": 4096,
        "rescale_factor": 0.00392156862745098,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
)

# The checkpoint's vision config; the tower's widths don't reach the inputs.
_VISION = MuseGlimmerVisionConfig(
    hidden_size=1536,
    intermediate_size=8960,
    num_hidden_layers=1,
    num_attention_heads=16,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    layer_types=["full_attention"],
    patch_size=14,
    patch_temporal=2,
    merge_size=2,
    pos_emb_height=32,
    pos_emb_width=32,
    rope_theta=10000.0,
    projector_hidden_size=4096,
)

# A small window clips at the edge of every image, and the default one only
# at the edge of large ones.
_WINDOWS = {
    "window32": _VISION,
    "window6": replace(_VISION, window_size_patches=6),
}


@pytest.fixture(scope="module", params=sorted(_WINDOWS))
def vision(request: pytest.FixtureRequest) -> MuseGlimmerVisionConfig:
    return _WINDOWS[request.param]


def _spec(vision: MuseGlimmerVisionConfig) -> GraphSpec:
    return GraphSpec(
        MuseGlimmerImagePreprocess(_PROCESSOR, vision, DType.bfloat16),
        "muse_glimmer_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )


@pytest.fixture(scope="module")
def preprocess(vision: MuseGlimmerVisionConfig) -> CompiledCallable[..., Any]:
    spec = _spec(vision)
    role = GraphRole.VISION_PREPROCESS
    session = InferenceSession(devices=[CPU()])
    return compile_and_init_graphs(session, {role: spec})[role]


_CASES: dict[str, Callable[[], Image.Image]] = {
    # Already a multiple of the merged patch, so the resize is the identity.
    "identity_560x420": lambda: noise(420, 560, "RGB"),
    # Every pixel differs, so any rounding difference in the resize shows.
    "noise_downscale_3000x2000": lambda: noise(2000, 3000, "RGB"),
    "round_501x333": lambda: scene(333, 501),
    # Over the token cap, so both sides shrink.
    "downscale_2400x1615": lambda: gradient(1615, 2400),
    "upscale_47x61": lambda: scene(61, 47),
    "tiny_9x13": lambda: gradient(9, 13),
    "extreme_1000x40": lambda: gradient(40, 1000),
    "rgba": lambda: scene(300, 202, "RGBA"),
    "gray": lambda: scene(200, 320, "L"),
    "gray_alpha": lambda: scene(260, 240, "LA"),
    "palette_transparency": palette_with_transparency,
    "chart_downscale_1920x2600": lambda: chart(2600, 1920),
    "chart_501x333": lambda: chart(333, 501),
}


def _served(
    image: Image.Image, vision: MuseGlimmerVisionConfig
) -> tuple[tuple[int, int], list[Buffer]]:
    """The image's resized size, and what MAX serve feeds the tower for it."""
    pixel_values, grid_thw = _PROCESSOR(image)
    _, grid_h, grid_w = (int(n) for n in grid_thw)
    tokens = grid_h * grid_w // _PROCESSOR.merge_size**2
    marker: Any = SimpleNamespace(
        pixel_values=pixel_values, start_idx=0, end_idx=tokens
    )
    context: Any = SimpleNamespace(
        request_id="request",
        images=[marker],
        extra_model_args={IMAGE_GRID_THW: grid_thw[None]},
    )
    packed = pack_uncached_images(
        [(context, [marker])], CPU(), vision, DType.bfloat16
    )
    assert packed is not None
    patch = _PROCESSOR.patch_size
    return (grid_h * patch, grid_w * patch), packed


@pytest.mark.parametrize("case", sorted(_CASES))
def test_matches_the_served_inputs(
    preprocess: CompiledCallable[..., Any],
    vision: MuseGlimmerVisionConfig,
    case: str,
) -> None:
    image = _CASES[case]()
    size, served = _served(image, vision)
    outputs = preprocess.execute_raw(*graph_inputs(image, size))
    check(preprocess, case, outputs, served)


def test_returns_the_vision_inputs_by_path() -> None:
    """A serving engine feeds the preprocess graph's outputs to the tower by
    path, so they are the tower's one-device inputs for an image."""
    outputs, out_tree = tree.flatten(_spec(_VISION).trace().output_tree)
    inputs, in_tree = tree.flatten(
        vision_input_tree(_VISION, DeviceRef.CPU(), DType.bfloat16)
    )
    assert out_tree.leaf_paths == in_tree.leaf_paths
    for output, want in zip(outputs, inputs, strict=True):
        assert (output.dtype, output.rank) == (want.dtype, want.rank)
        assert output.device == DeviceRef.CPU()
