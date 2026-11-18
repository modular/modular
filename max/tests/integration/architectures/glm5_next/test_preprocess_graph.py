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
"""Checks the GLM-5.3-Flash image preprocessing graph against MAX serve's.

A serving engine that runs the exported graph must feed the tower what MAX
serve does: the processor's patches and the batch processor's rotary
positions and segment bounds. Every output matches bit for bit, over images
the token budget leaves alone, scales up to the floor, and shrinks to the
cap, in every mode Pillow opens.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from max import tree
from max._artifact.vision import Glm5NextSmartResize, VisionMetadata
from max.driver import CPU, Buffer
from max.engine import InferenceSession
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.architectures.glm5_next.vision.geometry import (
    vision_cu_seqlens,
    vision_position_ids,
)
from max.pipelines.architectures.glm5_next.vision.image_processor import (
    Glm5NextImageProcessor,
)
from max.pipelines.architectures.glm5_next.vision.preprocess import (
    Glm5NextImagePreprocess,
    glm5_next_vision_metadata,
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

_PROCESSOR = Glm5NextImageProcessor()


@pytest.fixture(scope="module")
def preprocess() -> CompiledCallable[..., Any]:
    spec = GraphSpec(
        Glm5NextImagePreprocess(_PROCESSOR),
        "glm5_next_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )
    role = GraphRole.VISION_PREPROCESS
    session = InferenceSession(devices=[CPU()])
    return compile_and_init_graphs(session, {role: spec})[role]


def _served(image: Image.Image) -> list[Buffer]:
    """What MAX serve feeds the tower for one image, in the graph's order."""
    patches, grid = _PROCESSOR.preprocess(image)
    grids = np.array([grid], dtype=np.int32)
    row_offsets = vision_cu_seqlens(grids)
    max_seqlen = np.array(
        [np.diff(row_offsets.astype(np.int64)).max()], dtype=np.uint32
    )
    return [
        Buffer.from_numpy(patches),
        Buffer.from_numpy(vision_position_ids(grids, _PROCESSOR.merge_size)),
        Buffer.from_numpy(row_offsets),
        Buffer.from_numpy(max_seqlen),
    ]


def _canvas(image: Image.Image) -> tuple[int, int]:
    """The ``(height, width)`` the engine lays the image out at."""
    width, height = image.size
    return _PROCESSOR.canvas_size(height, width)


_CASES: dict[str, Callable[[], Image.Image]] = {
    # Within the budget: the canvas aligns up and the content keeps its size.
    "aligned 448x448": lambda: noise(448, 448, "RGB"),
    "unaligned 300x500": lambda: scene(300, 500, "RGB"),
    "tall strip": lambda: gradient(900, 60),
    # Under the token floor: the content scales up to it.
    "tiny 20x30": lambda: noise(20, 30, "RGB", seed=1),
    "thin 8x200": lambda: gradient(8, 200),
    # Over the token cap: the canvas shrinks by the binary search.
    "large 4000x3000": lambda: scene(4000, 3000, "RGB"),
    "panorama 1000x7000": lambda: gradient(1000, 7000),
    # Hard edges, which take a resize past [0, 255].
    "chart": lambda: chart(600, 800, seed=3),
    # Modes Pillow opens: the alpha is dropped, grey is converted.
    "rgba": lambda: noise(333, 517, "RGBA", seed=5),
    "grey": lambda: noise(257, 129, "L", seed=7),
    "grey alpha": lambda: scene(130, 310, "LA"),
    "palette transparency": palette_with_transparency,
}


@pytest.mark.parametrize("case", list(_CASES))
def test_image_graph_matches_max_serve(
    preprocess: CompiledCallable[..., Any], case: str
) -> None:
    image = _CASES[case]()
    outputs = preprocess(*graph_inputs(image, _canvas(image)))
    check(preprocess, case, tree.leaves(outputs), _served(image))


def test_metadata_records_the_processor_budget() -> None:
    config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=14),
        image_token_id=154854,
        image_start_token_id=154830,
        image_end_token_id=154831,
    )
    metadata = glm5_next_vision_metadata(config, _PROCESSOR)
    assert isinstance(metadata, VisionMetadata)
    image = metadata.config["image"]
    assert isinstance(image, dict)
    assert image["patch_size"] == 14 and image["merge_size"] == 2
    assert image["strategy"] == {
        "rule": Glm5NextSmartResize.rule,
        "temporal_patch_size": 2,
        "min_tokens": 16,
        "max_tokens": 8000,
        "max_entry_tokens": 0,
    }
    assert image["span"] == {
        "static": {
            "image_token_id": 154854,
            "begin_marker": None,
            "end_marker": None,
        }
    }
    assert image["template_markers"] == {"begin": 154830, "end": 154831}
    assert metadata.config["video"] is None


def test_a_text_only_checkpoint_has_no_metadata() -> None:
    config = SimpleNamespace(vision_config=None)
    assert glm5_next_vision_metadata(config, _PROCESSOR) is None
