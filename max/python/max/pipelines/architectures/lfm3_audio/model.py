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
"""The LFM3-Audio pipeline model: an audio graph and a language graph.

Clips reach the language graph through the pipeline's vision encoder cache,
as LFM3-VL's images do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from max import functional as F
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightData
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib import GraphSpec
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor import default_dtype

from ..lfm3_vl.lfm3_vl import Lfm3VlLanguage
from ..lfm3_vl.model import Lfm3MediaModel
from .lfm3_audio import Lfm3AudioLanguage, Lfm3AudioTower
from .model_config import Lfm3AudioConfig

_RENAMES = (
    ("encoder.lfm.lfm.", "encoder.lfm."),
    ("mlp.model.0.", "audio_projector.layer_norm."),
    ("mlp.model.1.", "audio_projector.linear_1."),
    ("mlp.model.3.", "audio_projector.linear_2."),
    ("conv.conv.weight", "conv.conv"),
)


def stride2_taps(steps: int, bins: int) -> npt.NDArray[np.int32]:
    """Returns the 3x3 stride-2 taps of a time-major (time, mel) plane.

    Row ``t * ceil(bins / 2) + m`` of the output lists the input rows under
    its window, mel offset major, or -1 outside the plane.
    """
    t = 2 * np.arange(-(steps // -2))[:, None, None, None]
    m = 2 * np.arange(-(bins // -2))[None, :, None, None]
    t = t + np.arange(3)[None, None, None, :] - 1
    m = m + np.arange(3)[None, None, :, None] - 1
    inside = (t >= 0) & (t < steps) & (m >= 0) & (m < bins)
    return np.where(inside, t * bins + m, -1).reshape(-1, 9).astype(np.int32)


def band_taps(n: int, window: int) -> npt.NDArray[np.int32]:
    """Returns each frame's ``window`` latest frames.

    Each window ends with the frame itself, and is -1 before the clip.
    """
    taps = np.arange(n)[:, None] - window + 1 + np.arange(window)[None]
    return np.where(taps >= 0, taps, -1).astype(np.int32)


@dataclass
class Lfm3AudioInputs:
    """One batch of clips for the audio graph."""

    features: Buffer
    taps_in: Buffer
    taps_mid: Buffer
    taps_end: Buffer
    position_ids: Buffer
    band_taps: Buffer


def pack_clips(
    clips: Sequence[npt.NDArray[np.float32]], window: int, device: Device
) -> Lfm3AudioInputs:
    """Packs clips, each ``[frames, mel bins]``, with offset taps."""
    columns: dict[str, list[npt.NDArray[Any]]] = {
        f.name: [] for f in fields(Lfm3AudioInputs)
    }
    rows = [0, 0, 0, 0]
    for clip in clips:
        steps, bins = clip.shape
        columns["features"].append(clip.reshape(-1, 1))
        for stage, name in enumerate(("taps_in", "taps_mid", "taps_end")):
            taps = stride2_taps(steps, bins)
            columns[name].append(np.where(taps >= 0, taps + rows[stage], -1))
            rows[stage] += steps * bins
            steps, bins = -(steps // -2), -(bins // -2)
        band = band_taps(steps, window)
        columns["position_ids"].append(np.arange(steps, dtype=np.int32))
        columns["band_taps"].append(np.where(band >= 0, band + rows[3], -1))
        rows[3] += steps
    return Lfm3AudioInputs(
        **{
            name: Buffer.from_numpy(np.ascontiguousarray(np.concatenate(v))).to(
                device
            )
            for name, v in columns.items()
        }
    )


class Lfm3AudioModel(Lfm3MediaModel[Lfm3AudioConfig]):
    """LFM3-Audio: the audio graph encodes clips the language graph reads."""

    model_config_cls: ClassVar[type[Any]] = Lfm3AudioConfig
    language_cls: ClassVar[type[Lfm3VlLanguage]] = Lfm3AudioLanguage

    def _load_state_dict(self) -> dict[str, WeightData]:
        audio: dict[str, WeightData] = {}
        language: dict[str, WeightData] = {}
        for name, weight in self.weights.items():
            if name.startswith("lfm."):
                language[name.removeprefix("lfm.")] = weight.data()
            elif name.startswith(("encoder.", "mlp.")):
                for old, new in _RENAMES:
                    name = name.replace(old, new)
                audio[name] = weight.data()
        self._vision_weights_dict = audio
        self._language_weights_dict = language
        return {**audio, **language}

    def _vision_spec(
        self, model_config: Lfm3AudioConfig, state_dict: dict[str, Any]
    ) -> GraphSpec:
        with F.lazy(), default_dtype(model_config.dtype):
            tower = Lfm3AudioTower(model_config)
            tower.to(self.devices[0])
        device = DeviceRef.from_device(self.devices[0])
        window = model_config.encoder.sliding_window

        def taps(rows: str, width: int) -> TensorType:
            return TensorType(DType.int32, [rows, width], device=device)

        return GraphSpec(
            tower,
            type(tower).__qualname__,
            [
                TensorType(DType.float32, ["rows_0", 1], device=device),
                taps("rows_1", 9),
                taps("rows_2", 9),
                taps("rows_3", 9),
                TensorType(DType.int32, ["num_tokens"], device=device),
                taps("num_tokens", window),
            ],
            weights=state_dict,
        )

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> object | None:
        """Packs the batch's audio clips for the audio graph."""
        clips = [clip.pixel_values for _, clips in selection for clip in clips]
        if not clips:
            return None
        return pack_clips(
            clips, self._config.encoder.sliding_window, devices[0]
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: object | None,
    ) -> VisionEncodeResult:
        """Encodes the batch's audio clips into embeddings."""
        packed = packed or self.pack_vision_inputs(selection, devices)
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert isinstance(packed, Lfm3AudioInputs)
        assert self.vision_model is not None
        embeddings = self.vision_model(
            *(getattr(packed, f.name) for f in fields(packed))
        )
        return VisionEncodeResult(embeddings=[embeddings.driver_tensor])
