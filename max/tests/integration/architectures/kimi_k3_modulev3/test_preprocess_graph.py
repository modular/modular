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
"""Checks the Kimi K3 image preprocessing graph against host processing.

The graph re-expresses Pillow's RGBA resize (premultiply, resize,
unpremultiply, and the same-size copy) and the checkpoint's compositing as
lookup tables. Each table can be right alone while the composition is wrong,
so the exported graph must match the host processor bit for bit, including
for partial alpha at a resized size.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Buffer, float32_array_to_buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.architectures.kimi_k3_modulev3.vision_preprocessing import (
    KimiK3ImagePreprocess,
)
from max.pipelines.architectures.kimi_k3_modulev3.vision_processor import (
    KimiK3VisionProcessor,
    MediaProcConfig,
    compute_position_ids,
    navit_resize_image,
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
    graph_inputs,
    noise,
    scene,
)

# The checkpoint's transparency settings, with a small budget that reaches
# the shrink path without large images.
_CONFIG = MediaProcConfig.from_dict(
    {
        "in_patch_limit": 1024,
        "transparent_bg_config": {
            "pattern": "chessboard",
            "chessboard_square_size": 8,
            "chessboard_square_on_top_left": True,
            "chessboard_white_value": 255,
            "chessboard_gray_value": 180,
        },
        "transparent_bg_fill_stage": "after_resize",
    }
)


@pytest.fixture(
    scope="module",
    params=[
        _CONFIG,
        replace(_CONFIG, transparent_bg_fill_stage="before_resize"),
        replace(
            _CONFIG,
            transparent_bg_fill_stage="before_resize",
            transparent_pattern=None,
        ),
    ],
    ids=["after resize", "before resize", "default transparency"],
)
def config(request: pytest.FixtureRequest) -> MediaProcConfig:
    assert isinstance(request.param, MediaProcConfig)
    return request.param


@pytest.fixture(scope="module")
def preprocess(config: MediaProcConfig) -> CompiledCallable[..., Any]:
    spec = GraphSpec(
        KimiK3ImagePreprocess(config),
        "kimi_k3_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )
    role = GraphRole.VISION_PREPROCESS
    session = InferenceSession(devices=[CPU()])
    return compile_and_init_graphs(session, {role: spec})[role]


def _served(image: Image.Image, config: MediaProcConfig) -> list[Buffer]:
    """What host processing feeds the tower for one image, in graph order."""
    output = KimiK3VisionProcessor(config).preprocess(
        [{"type": "image", "image": image}]
    )
    grids = output["grid_thws"]
    count = int(np.prod(grids, axis=1).sum())
    return [
        float32_array_to_buffer(
            output["pixel_values"], dtype=DType.bfloat16, device=CPU()
        ),
        Buffer.from_numpy(compute_position_ids(grids)),
        Buffer.from_numpy(grids),
        Buffer.from_numpy(np.array([0, count], dtype=np.uint32)),
        Buffer.from_numpy(np.array([count], dtype=np.uint32)),
    ]


def _canvas(image: Image.Image, config: MediaProcConfig) -> tuple[int, int]:
    """The padded ``(height, width)`` a serving engine passes as ``size``."""
    plan = navit_resize_image(*image.size, config)
    return (
        plan["new_height"] + plan["pad_height"],
        plan["new_width"] + plan["pad_width"],
    )


_CASES: dict[str, Callable[[], Image.Image]] = {
    # Within the budget: Pillow copies instead of resizing.
    "native rgb": lambda: noise(100, 130, "RGB"),
    "native rgba": lambda: noise(90, 70, "RGBA", seed=1),
    # Over the budget: the content shrinks, then pads to merged patches.
    "shrunk rgb": lambda: scene(700, 500, "RGB"),
    "shrunk rgba": lambda: scene(640, 610, "RGBA"),
    "shrunk noisy rgba": lambda: noise(533, 611, "RGBA", seed=2),
    "chart": lambda: chart(600, 800, seed=3),
    "grey": lambda: noise(900, 257, "L", seed=4),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_image_graph_matches_host_processing(
    preprocess: CompiledCallable[..., Any], config: MediaProcConfig, case: str
) -> None:
    image = _CASES[case]()
    outputs = preprocess.execute_raw(
        *graph_inputs(image, _canvas(image, config))
    )
    check(preprocess, case, outputs, _served(image, config))
