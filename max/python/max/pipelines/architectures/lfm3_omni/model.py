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
"""The LFM3-Omni pipeline model.

LFM3-VL's vision graph, LFM3-Audio's audio graph and one language graph reading
both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.graph.weights import WeightData
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib import GraphSpec
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult

from ..lfm3_audio.model import (
    Lfm3AudioInputs,
    Lfm3SpeechModel,
    audio_tower_spec,
    pack_clips,
)
from ..lfm3_audio.model_config import Lfm3AudioEncoderConfig
from ..lfm3_vl.model import Lfm3VlModel, Lfm3VlVisionInputs
from ..lfm3_vl.model_config import Lfm3VlConfig
from .model_config import Lfm3OmniConfig

_AUDIO = "audio"

Selection = Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]]


def _is_clip(item: ImageMetadata) -> bool:
    """Whether an encoder input is an audio clip.

    A clip is ``[frames, mel bins]``, unlike a frame's patches.
    """
    return item.pixel_values.ndim == 2


def _of_kind(selection: Selection, clips: bool) -> Selection:
    return [
        (ctx, [item for item in items if _is_clip(item) == clips])
        for ctx, items in selection
    ]


class Lfm3ListenerModel(Lfm3VlModel):
    """A vision model that also hears when :attr:`audio_encoder` is set.

    Frames and clips each run through their own tower, and their rows reach the
    language graph in placeholder order.
    """

    @property
    def audio_encoder(self) -> Lfm3AudioEncoderConfig | None:
        """The audio encoder's config, if the model takes audio."""
        raise NotImplementedError

    def _load_state_dict(self) -> dict[str, WeightData]:
        state = super()._load_state_dict()
        audio: dict[str, WeightData] = {}
        for name, weight in self.weights.items():
            name = name.removeprefix("model.")
            if name.startswith(("audio_encoder.", "audio_projector.")):
                name = name.replace("audio_encoder.", "encoder.", 1)
                audio[name.replace("conv.conv.weight", "conv.conv")] = (
                    weight.data()
                )
        self._audio_weights_dict = audio
        return {**state, **audio}

    def graph_specs(self, model_config: Lfm3VlConfig) -> dict[str, GraphSpec]:
        """Returns the graph specs, with the audio graph if the model hears."""
        specs = super().graph_specs(model_config)
        if (encoder := self.audio_encoder) is not None:
            specs[_AUDIO] = audio_tower_spec(
                model_config,
                encoder,
                self._audio_weights_dict,
                self.devices[0],
            )
        return specs

    def pack_vision_inputs(
        self, selection: Selection, devices: list[Device]
    ) -> object | None:
        """Packs the frames and the clips for their own towers."""
        frames = super().pack_vision_inputs(_of_kind(selection, False), devices)
        clips = [
            item.pixel_values
            for _, items in _of_kind(selection, True)
            for item in items
        ]
        if not clips:
            return None if frames is None else (frames, None)
        if (encoder := self.audio_encoder) is None:
            raise ValueError("This model doesn't take audio.")
        return frames, pack_clips(clips, encoder.sliding_window, devices[0])

    def vision_execute(
        self,
        selection: Selection,
        devices: list[Device],
        packed: object | None,
    ) -> VisionEncodeResult:
        """Encodes frames and clips into rows in placeholder order."""
        packed = packed or self.pack_vision_inputs(selection, devices)
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert isinstance(packed, tuple)
        frames, clips = packed
        rows: dict[bool, Buffer] = {}
        if frames is not None:
            assert isinstance(frames, Lfm3VlVisionInputs)
            rows[False] = (
                super()
                .vision_execute(_of_kind(selection, False), devices, frames)
                .embeddings[0]
            )
        if clips is not None:
            assert isinstance(clips, Lfm3AudioInputs)
            rows[True] = self.compiled_modules[_AUDIO](
                *(getattr(clips, f.name) for f in fields(clips))
            ).driver_tensor
        if len(rows) == 1:
            return VisionEncodeResult(embeddings=list(rows.values()))
        items = [item for _, items in selection for item in items]
        out = Buffer(
            self._config.dtype,
            [
                sum(item.embedding_rows for item in items),
                self._config.hidden_size,
            ],
            device=devices[0],
        )
        start, taken = 0, {False: 0, True: 0}
        for item in items:
            kind, n = _is_clip(item), item.embedding_rows
            out[start : start + n, :].inplace_copy_from(
                rows[kind][taken[kind] : taken[kind] + n, :]
            )
            start += n
            taken[kind] += n
        return VisionEncodeResult(embeddings=[out])


class Lfm3OmniModel(Lfm3SpeechModel[Lfm3OmniConfig], Lfm3ListenerModel):
    """LFM3-Omni: a listening vision model that speaks."""

    model_config_cls: ClassVar[type[Any]] = Lfm3OmniConfig
    _config: Lfm3OmniConfig

    @property
    def audio_encoder(self) -> Lfm3AudioEncoderConfig:
        """The speech encoder's config."""
        return self._config.encoder

    def graph_specs(self, model_config: Lfm3VlConfig) -> dict[str, GraphSpec]:
        """Returns the vision, audio and language graph specs."""
        assert isinstance(model_config, Lfm3OmniConfig)
        return super().graph_specs(model_config)

    def _load_state_dict(self) -> dict[str, WeightData]:
        state = super()._load_state_dict()
        for name, weight in self.weights.items():
            name = name.removeprefix("model.")
            if name.startswith("codebook_embedding."):
                self._language_weights_dict[name] = weight.data()
        return state
