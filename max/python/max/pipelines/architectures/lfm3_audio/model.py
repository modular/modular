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
"""The LFM3-Audio pipeline model.

An audio graph, a language graph, and the depth and codec graphs that speak.

Clips reach the language graph through the pipeline's vision encoder cache,
as LFM3-VL's images do.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import huggingface_hub
import numpy as np
import numpy.typing as npt
from max import functional as F
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import SafetensorWeights, WeightData
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
)
from max.pipelines.lib import GraphSpec, ModelInputs
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.pipelines.weights import HuggingFaceRepo
from max.tensor import default_dtype

from ..lfm3_vl.lfm3_vl import Lfm3VlLanguage
from ..lfm3_vl.model import Lfm3MediaModel, Lfm3VlInputs
from .lfm3_audio import Lfm3AudioTower, Lfm3SpeechLanguage
from .model_config import CODEC_SUBFOLDER, Lfm3AudioConfig
from .speech import (
    Lfm3CodecDecoder,
    Lfm3DepthDecoder,
    codec_decoder_weights,
    depth_decoder_weights,
)

_DEPTH = "speech_depth"
_CODEC = "speech_codec"

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


def audio_tower_spec(
    config: Lfm3AudioConfig, state_dict: dict[str, Any], device: Device
) -> GraphSpec:
    """Returns the graph of the speech encoder and projector.

    Its inputs are the ones :func:`pack_clips` packs.
    """
    with F.lazy(), default_dtype(config.dtype):
        tower = Lfm3AudioTower(config)
        tower.to(device)
    ref = DeviceRef.from_device(device)

    def taps(rows: str, width: int) -> TensorType:
        return TensorType(DType.int32, [rows, width], device=ref)

    return GraphSpec(
        tower,
        type(tower).__qualname__,
        [
            TensorType(DType.float32, ["rows_0", 1], device=ref),
            taps("rows_1", 9),
            taps("rows_2", 9),
            taps("rows_3", 9),
            TensorType(DType.int32, ["num_tokens"], device=ref),
            taps("num_tokens", config.encoder.sliding_window),
        ],
        weights=state_dict,
    )


def codec_weights(repo: HuggingFaceRepo) -> SafetensorWeights:
    """Returns the codec's weights.

    A checkpoint keeps them in their own folder.
    """
    name = f"{CODEC_SUBFOLDER}/model.safetensors"
    if repo.repo_type == "local":
        path = os.path.join(repo.local_path, name)
    else:
        path = huggingface_hub.hf_hub_download(
            repo.repo_id, name, revision=repo.revision
        )
    return SafetensorWeights([Path(path)])


_SpeechConfigT = TypeVar("_SpeechConfigT", bound=Lfm3AudioConfig)


class Lfm3SpeechModel(Lfm3MediaModel[_SpeechConfigT]):
    """An LFM3 media model with the depth decoder and codec that speak.

    The language graph reads each spoken frame as the sum of its codes'
    embeddings at the frame's placeholder. The engine that serves speech
    samples the frames with the depth decoder and decodes them with the
    codec; this pipeline feeds the language graph no frames.
    """

    language_cls: ClassVar[type[Lfm3VlLanguage]] = Lfm3SpeechLanguage

    def graph_specs(self, model_config: _SpeechConfigT) -> dict[str, GraphSpec]:
        """Returns the graph specs, adding the depth decoder and the codec."""
        specs = super().graph_specs(model_config)
        device = self.devices[0]
        ref = DeviceRef.from_device(device)
        c = model_config.speech.num_codebooks
        codec = model_config.codec
        with F.lazy(), default_dtype(model_config.dtype):
            depth = Lfm3DepthDecoder(model_config)
            depth.to(device)
            decoder = Lfm3CodecDecoder(codec)
            decoder.to(device)
        specs[_DEPTH] = GraphSpec(
            depth,
            type(depth).__qualname__,
            [
                TensorType(
                    model_config.dtype,
                    ["batch", model_config.hidden_size],
                    device=ref,
                ),
                TensorType(DType.int32, ["frames"], device=ref),
                TensorType(DType.int32, ["frames", c], device=ref),
                TensorType(DType.bool, ["frames", c], device=ref),
                TensorType(DType.int32, ["slots"], device=ref),
                TensorType(DType.int32, ["slots", c], device=ref),
            ],
            weights=depth_decoder_weights(self.weights),
        )
        specs[_CODEC] = GraphSpec(
            decoder,
            type(decoder).__qualname__,
            [
                TensorType(
                    DType.float32,
                    ["frames", c * len(codec.fsq_levels)],
                    device=ref,
                ),
                TensorType(DType.int32, ["rows"], device=ref),
                TensorType(
                    DType.int32,
                    ["rows", codec.decoder.sliding_window],
                    device=ref,
                ),
            ],
            weights=codec_decoder_weights(
                codec_weights(self.pipeline_config.model.huggingface_model_repo)
            ),
        )
        return specs

    def _language_input_types(
        self, model_config: _SpeechConfigT
    ) -> list[TensorType | BufferType]:
        types = super()._language_input_types(model_config)
        ref = DeviceRef.from_device(self.devices[0])
        frames = [
            TensorType(
                DType.int32,
                ["num_frames", model_config.speech.num_codebooks],
                device=ref,
            ),
            TensorType(DType.int32, ["num_frames"], device=ref),
        ]
        # After the media inputs, ahead of the cache inputs.
        return [*types[:5], *frames, *types[5:]]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Adds an empty batch of spoken frames."""
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        assert isinstance(inputs, Lfm3VlInputs)
        device = self.devices[0]
        c = self._config.speech.num_codebooks
        inputs.speech = (
            Buffer.zeros(shape=[0, c], dtype=DType.int32).to(device),
            Buffer.zeros(shape=[0], dtype=DType.int32).to(device),
        )
        return inputs


class Lfm3AudioModel(Lfm3SpeechModel[Lfm3AudioConfig]):
    """LFM3-Audio: the audio graph encodes clips the language graph reads."""

    model_config_cls: ClassVar[type[Any]] = Lfm3AudioConfig

    def _load_state_dict(self) -> dict[str, WeightData]:
        audio: dict[str, WeightData] = {}
        language: dict[str, WeightData] = {}
        for name, weight in self.weights.items():
            if name.startswith("lfm."):
                language[name.removeprefix("lfm.")] = weight.data()
            elif name.startswith("codebook_embedding."):
                language[name] = weight.data()
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
        return audio_tower_spec(model_config, state_dict, self.devices[0])

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
