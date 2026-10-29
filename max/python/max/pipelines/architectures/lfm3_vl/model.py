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
"""The LFM3-VL pipeline model: a vision graph and a language graph.

Images reach the language graph through the pipeline's vision encoder
cache, which this model packs and encodes for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
from max import functional as F
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.legacy.kv_cache import KVCacheInputs
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.context import ImageMetadata, TextAndVisionContext
from max.pipelines.lib import (
    GraphSpec,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_gemma_multimodal_language_symbolic_inputs,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.tensor import default_dtype

from ..lfm3.model_config import Lfm3Config
from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from ..llama3_modulev3.inputs import Llama3Inputs
from .lfm3_vl import Lfm3VlLanguage, Lfm3VlVision
from .model_config import Lfm3VlConfig


@dataclass
class Lfm3VlInputs(Llama3Inputs):
    """The decoder inputs, with the image embeddings after the row offsets."""

    speech: tuple[Buffer, ...] = ()
    """A speaking model's frame codes and their placeholder indices."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The graph inputs, with the media inputs after the row offsets."""
        base = super().buffers
        return (
            *base[:3],
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            *self.speech,
            *base[3:],
        )


class Lfm3VlBatchProcessor(Llama3ModuleV3BatchProcessor):
    """Ragged batching whose inputs carry the image embeddings."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> Lfm3VlInputs:
        return Lfm3VlInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )


@dataclass
class Lfm3VlVisionInputs:
    """One batch of frames for the vision graph."""

    patches: Buffer
    position_ids: Buffer
    cu_seqlens: Buffer
    max_seqlen: Buffer
    conv_taps: Buffer


def frame_layout(
    blocks_h: int, blocks_w: int, merge: int, kernel: int
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Returns a frame's patch positions and convolution taps.

    Patches are in merge-block order. Each row of the positions is a patch's
    ``(row, col)`` on the grid; each row of the taps lists the patches under
    its ``kernel x kernel`` window in raster order, or -1 off the grid.
    """
    h, w = blocks_h * merge, blocks_w * merge
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r, c = (
        a.reshape(blocks_h, merge, blocks_w, merge).transpose(0, 2, 1, 3)
        for a in (rows, cols)
    )
    r, c = r.reshape(-1), c.reshape(-1)
    order = np.empty((h, w), dtype=np.int32)
    order[r, c] = np.arange(h * w, dtype=np.int32)
    offsets = np.arange(kernel) - kernel // 2
    rr = r[:, None, None] + offsets[None, :, None]
    cc = c[:, None, None] + offsets[None, None, :]
    inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    taps = np.where(
        inside, order[rr.clip(0, h - 1), cc.clip(0, w - 1)], -1
    ).reshape(h * w, kernel * kernel)
    return np.stack([r, c], axis=-1).astype(np.int32), taps.astype(np.int32)


def pack_frames(
    frames: Sequence[ImageMetadata], merge: int, kernel: int, device: Device
) -> Lfm3VlVisionInputs:
    """Packs frames, each ``[blocks_h, blocks_w, merge**2, patch_dim]``."""
    patches: list[npt.NDArray[np.float32]] = []
    positions: list[npt.NDArray[np.int32]] = []
    taps: list[npt.NDArray[np.int32]] = []
    lengths: list[int] = []
    for frame in frames:
        blocks_h, blocks_w, _, patch_dim = frame.pixel_values.shape
        position, tap = frame_layout(blocks_h, blocks_w, merge, kernel)
        tap[tap >= 0] += sum(lengths)
        patches.append(frame.pixel_values.reshape(-1, patch_dim))
        positions.append(position)
        taps.append(tap)
        lengths.append(len(position))
    all_taps = np.concatenate(taps)
    all_taps[all_taps < 0] = sum(lengths)
    cu_seqlens = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)

    def to_device(a: npt.NDArray[Any]) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(a)).to(device)

    return Lfm3VlVisionInputs(
        patches=to_device(np.concatenate(patches).astype(np.float32)),
        position_ids=to_device(np.concatenate(positions)),
        cu_seqlens=to_device(cu_seqlens),
        max_seqlen=Buffer.from_numpy(np.array(max(lengths), dtype=np.uint32)),
        conv_taps=to_device(all_taps),
    )


_ConfigT = TypeVar("_ConfigT", bound=Lfm3Config)


class Lfm3MediaModel(
    LogProbabilitiesMixin,
    ModuleV3MultiGraphPipelineModelWithKVCache[TextAndVisionContext, _ConfigT],
):
    """An LFM3 media model.

    A media graph encodes what the language graph reads at its placeholders.
    """

    batch_processor_cls: ClassVar[type[Lfm3VlBatchProcessor]] = (
        Lfm3VlBatchProcessor
    )
    language_cls: ClassVar[type[Lfm3VlLanguage]]
    _config: _ConfigT

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        if len(self.devices) > 1:
            raise NotImplementedError("LFM3 media models run on one device.")
        self.vision_model, self.language_model = self._load_towers(session)
        self._empty_embeddings = [
            Buffer.zeros(
                shape=[0, self._config.hidden_size], dtype=self._config.dtype
            ).to(self.devices[0])
        ]

    @property
    def model(self) -> Callable[..., Any]:
        """The language tower, which device graph capture replays."""
        return self.language_model

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> _ConfigT:
        del state_dict
        assert self.model_config_cls is not None
        config = self.model_config_cls.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.return_logits = self.return_logits
        self._config = config
        return config

    def _language_spec(
        self, model_config: _ConfigT, state_dict: dict[str, Any]
    ) -> GraphSpec:
        with F.lazy(), default_dtype(model_config.dtype):
            language = self.language_cls(model_config)
            language.to(self.devices[0])
        return GraphSpec(
            language,
            type(language).__qualname__,
            self._language_input_types(model_config),
            weights=state_dict,
        )

    def _language_input_types(
        self, model_config: _ConfigT
    ) -> list[TensorType | BufferType]:
        """Returns the graph inputs in :class:`Lfm3VlInputs` order.

        The tokens, logit count and row offsets, the media inputs, then the
        cache inputs.
        """
        return modulev3_gemma_multimodal_language_symbolic_inputs(
            kv_params=self.kv_params,
            device_ref=DeviceRef.from_device(self.devices[0]),
            hidden_size=model_config.hidden_size,
            embedding_dtype=model_config.dtype,
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns the embeddings of a batch without media."""
        return self._empty_embeddings

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the language graph on the inputs."""
        outputs = self.language_model(*model_inputs.buffers)
        logits = outputs.logits
        if logits is None:
            logits = outputs.next_token_logits
        return ModelOutputs(
            logits=logits.driver_tensor,
            next_token_logits=outputs.next_token_logits.driver_tensor,
            logit_offsets=(
                None
                if outputs.logit_offsets is None
                else outputs.logit_offsets.driver_tensor
            ),
            hidden_states=(
                None
                if outputs.hidden_states is None
                else outputs.hidden_states.driver_tensor
            ),
        )


class Lfm3VlModel(Lfm3MediaModel[Lfm3VlConfig]):
    """LFM3-VL: the vision graph encodes frames the language graph reads."""

    model_config_cls: ClassVar[type[Any]] = Lfm3VlConfig
    language_cls: ClassVar[type[Lfm3VlLanguage]] = Lfm3VlLanguage

    def _load_state_dict(self) -> dict[str, WeightData]:
        vision: dict[str, WeightData] = {}
        language: dict[str, WeightData] = {}
        for name, weight in self.weights.items():
            name = name.removeprefix("model.")
            if name.startswith("language_model."):
                language["model." + name.removeprefix("language_model.")] = (
                    weight.data()
                )
            elif name.startswith(("vision_tower.", "multi_modal_projector.")):
                name = name.replace("spatial_conv.weight", "spatial_conv")
                vision[name] = weight.data()
        self._vision_weights_dict = vision
        self._language_weights_dict = language
        return {**vision, **language}

    def _vision_spec(
        self, model_config: Lfm3VlConfig, state_dict: dict[str, Any]
    ) -> GraphSpec:
        with F.lazy(), default_dtype(model_config.dtype):
            vision = Lfm3VlVision(model_config)
            vision.to(self.devices[0])
        device = DeviceRef.from_device(self.devices[0])
        vision_config = model_config.vision
        return GraphSpec(
            vision,
            type(vision).__qualname__,
            [
                TensorType(
                    DType.float32,
                    ["num_patches", vision_config.patch_dim],
                    device=device,
                ),
                TensorType(DType.int32, ["num_patches", 2], device=device),
                TensorType(DType.uint32, ["num_frames_plus_1"], device=device),
                TensorType(DType.uint32, [], device=DeviceRef.CPU()),
                TensorType(
                    DType.int32,
                    ["num_patches", vision_config.conv_kernel_size**2],
                    device=device,
                ),
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
        """Packs the batch's frames for the vision graph."""
        frames = [frame for _, frames in selection for frame in frames]
        if not frames:
            return None
        vision = self._config.vision
        return pack_frames(
            frames, vision.merge_size, vision.conv_kernel_size, devices[0]
        )

    def vision_execute(
        self,
        selection: Sequence[
            tuple[TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: object | None,
    ) -> VisionEncodeResult:
        """Encodes the batch's frames into embeddings."""
        packed = packed or self.pack_vision_inputs(selection, devices)
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert isinstance(packed, Lfm3VlVisionInputs)
        assert self.vision_model is not None
        embeddings = self.vision_model(
            packed.patches,
            packed.position_ids,
            packed.cu_seqlens,
            packed.max_seqlen,
            packed.conv_taps,
        )
        return VisionEncodeResult(embeddings=[embeddings.driver_tensor])
