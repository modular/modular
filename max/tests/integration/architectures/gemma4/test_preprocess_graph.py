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
"""Checks the Gemma 4 image preprocessing graph against MAX serve's.

A serving engine that runs the exported graph must feed the tower what MAX
serve does: the image processor's patches and positions, and the segment
bounds and pooling table the model packs for them. Every output but the
patches matches bit for bit. The graph resizes uint8 pixels, accumulating
in float32 and rounding once at the end, where the processor resizes with
Pillow, which rounds and clamps between its two passes, so the patches
match within a level or two, and further around a chart's hard edges.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    pack_vision_buffers,
)
from max.pipelines.architectures.gemma4.image_processor import (
    Gemma4ImageProcessor,
)
from max.pipelines.architectures.gemma4.model import Gemma3_MultiModalModel
from max.pipelines.architectures.gemma4.video_processor import (
    Gemma4VideoProcessor,
    VideoMetadata,
)
from max.pipelines.architectures.gemma4.vision_model.preprocess_graph import (
    Gemma4ImagePreprocess,
)
from max.pipelines.architectures.gemma4.vision_model.vision_model import (
    Gemma4VisionModel,
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

_PATCH = 16
_K = 3


def _compile(processor: Gemma4ImageProcessor) -> CompiledCallable[..., Any]:
    spec = GraphSpec(
        Gemma4ImagePreprocess(processor, DType.bfloat16),
        "gemma4_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )
    role = GraphRole.VISION_PREPROCESS
    session = InferenceSession(devices=[CPU()])
    return compile_and_init_graphs(session, {role: spec})[role]


@pytest.fixture(scope="module")
def preprocess() -> CompiledCallable[..., Any]:
    return _compile(Gemma4ImageProcessor())


_CASES: dict[str, Callable[[], Image.Image]] = {
    # The resize is skipped at the target size.
    "identity_768x768": lambda: noise(768, 768, "RGB"),
    "downscale_3000x2000": lambda: scene(2000, 3000),
    "downscale_1500x1500": lambda: gradient(1500, 1500),
    "portrait_768x1024": lambda: scene(1024, 768),
    "upscale_96x96": lambda: scene(96, 96),
    "tiny_9x13": lambda: gradient(9, 13),
    # The short side rounds to zero, so it takes one block.
    "extreme_10x3000": lambda: gradient(10, 3000),
    "rgba": lambda: scene(150, 101, "RGBA"),
    "gray": lambda: scene(100, 160, "L"),
    "gray_alpha": lambda: scene(130, 120, "LA"),
    "palette_transparency": palette_with_transparency,
}

_CHARTS: dict[str, Callable[[], Image.Image]] = {
    "downscale_3000x2000": lambda: chart(2000, 3000),
    "upscale_96x96": lambda: chart(96, 96),
}


def _level(processor: Gemma4ImageProcessor) -> npt.NDArray[np.float32]:
    """How far one uint8 level moves each value of a patch row, whose
    values run over the pixels with the channels innermost."""
    step = np.full(3, processor.rescale_factor if processor.do_rescale else 1.0)
    if processor.do_normalize:
        step = step / processor.image_std
    return np.tile(step, _PATCH * _PATCH).astype(np.float32)


def _served(
    image: Image.Image, processor: Gemma4ImageProcessor | None = None
) -> tuple[tuple[int, int], list[Buffer]]:
    """What MAX serve feeds the tower for the image, and its resized size."""
    processor = processor or Gemma4ImageProcessor()
    (patches,), (positions,), (soft_tokens,) = processor([image])
    return _packed(patches, positions, soft_tokens)


def _packed(
    patches: npt.NDArray[np.float32],
    positions: npt.NDArray[np.int32],
    soft_tokens: int,
) -> tuple[tuple[int, int], list[Buffer]]:
    """What MAX serve feeds the tower for one item's patches, and the item's
    resized size."""
    patches_w, patches_h = (int(n) + 1 for n in positions.max(axis=0))
    packed = pack_vision_buffers(
        [CPU()],
        _K,
        [patches],
        [positions],
        [len(patches)],
        [soft_tokens],
        DType.bfloat16,
    )
    served = [
        packed.patches_flat[0],
        packed.pixel_position_ids[0],
        packed.cu_seqlens[0],
        packed.pool_gather_index[0],
        packed.max_seq_len,
    ]
    return (patches_h * _PATCH, patches_w * _PATCH), served


@pytest.mark.parametrize("case", sorted(_CASES))
def test_matches_the_served_inputs(
    preprocess: CompiledCallable[..., Any], case: str
) -> None:
    image = _CASES[case]()
    size, served = _served(image)
    outputs = preprocess.execute_raw(*graph_inputs(image, size))
    check(preprocess, case, outputs, served, _level(Gemma4ImageProcessor()))


@pytest.mark.parametrize("case", sorted(_CHARTS))
def test_a_chart_matches_the_served_inputs(
    preprocess: CompiledCallable[..., Any], case: str
) -> None:
    """A chart's text, lines and bars, whose hard edges near black and white
    Pillow clamps between its passes, match within the hard-edge bounds."""
    image = _CHARTS[case]()
    size, served = _served(image)
    outputs = preprocess.execute_raw(*graph_inputs(image, size))
    check(
        preprocess,
        f"chart_{case}",
        outputs,
        served,
        _level(Gemma4ImageProcessor()),
        hard_edges=True,
    )


def test_matches_a_normalizing_processor() -> None:
    """A processor that normalizes with a non-unit std normalizes each level
    as numpy does, whose float32 quotient the graph's division can miss by
    an ulp."""
    processor = Gemma4ImageProcessor(
        do_normalize=True,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
    )
    image = gradient(300, 451)
    size, served = _served(image, processor)
    graph = _compile(processor)
    check(
        graph,
        "normalized",
        graph.execute_raw(*graph_inputs(image, size)),
        served,
        _level(processor),
    )


def test_a_video_frame_is_a_one_frame_image(
    preprocess: CompiledCallable[..., Any],
) -> None:
    """Gemma 4 exports no video graph: the image graph, given a frame and
    the video processor's smaller target, gives the tower what MAX serve's
    video processor makes of that frame."""
    frames = [scene(720 + i, 1280).crop((0, 0, 1280, 720)) for i in range(3)]
    processor = Gemma4VideoProcessor()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            processor, "_decode_video", lambda _: (frames, VideoMetadata())
        )
        (pixel_values,), (position_ids,), (soft_tokens,), _ = processor([b""])
    for i, frame in enumerate(frames):
        real = position_ids[i, :, 0] >= 0
        size, served = _packed(
            pixel_values[i][real], position_ids[i][real], soft_tokens
        )
        outputs = preprocess.execute_raw(*graph_inputs(frame, size))
        check(
            preprocess,
            f"frame {i}",
            outputs,
            served,
            _level(Gemma4ImageProcessor()),
        )


def _model(tmp_path: Path, vision_config: object) -> Any:
    """A stand-in with what ``vision_preprocess_spec`` reads."""
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"image_processor": {"patch_size": _PATCH}})
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            vision_config=vision_config, unquantized_dtype=DType.bfloat16
        ),
        pipeline_config=SimpleNamespace(
            model=SimpleNamespace(
                model_path=str(tmp_path), huggingface_model_revision=None
            )
        ),
    )


def test_vision_preprocess_returns_the_vision_inputs_by_path(
    tmp_path: Path,
) -> None:
    """A serving engine feeds the preprocess graph's outputs to the tower by
    path, so they are the tower's one-device inputs for an image."""
    model = _model(tmp_path, SimpleNamespace(patch_size=_PATCH))
    spec = Gemma3_MultiModalModel.vision_preprocess_spec(model)
    assert spec is not None
    outputs, out_tree = tree.flatten(spec.trace().output_tree)
    tower: Any = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=_PATCH),
            devices=[DeviceRef.CPU()],
            unquantized_dtype=DType.bfloat16,
        )
    )
    inputs, in_tree = tree.flatten(Gemma4VisionModel.input_tree(tower))
    assert out_tree.leaf_paths == in_tree.leaf_paths
    for output, want in zip(outputs, inputs, strict=True):
        assert (output.dtype, output.rank) == (want.dtype, want.rank)
        assert output.device == DeviceRef.CPU()


def test_a_checkpoint_without_a_tower_has_no_preprocess_graph(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, None)
    assert Gemma3_MultiModalModel.vision_preprocess_spec(model) is None
