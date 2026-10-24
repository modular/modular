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
"""The GLM-5.3-Flash (``glm5_next``) pipeline model.

Derives from the DeepSeek-V3.2 pipeline model, which already owns the sparse
MLA, DSA indexer, MoE and expert-parallel plumbing GLM-5.3-Flash inherits
through GLM-5.2. What this class adds is everything the hybrid schedule
implies: a second per-request pool for the KDA conv and recurrent state, which
the framework's KV accounting does not see, and a readiness check that names
the bring-up lanes still outstanding.

Kept deliberately thin. Where GLM-5.3-Flash needs behaviour DeepSeek-V3.2 does
not have, the fix belongs in the layer or the config, not here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, Usage
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef, TensorType, TensorValue
from max.graph.weights import WeightData
from max.nn.kv_cache.cache_params import KVCacheParams, MultiKVCacheParams
from max.nn.layer import LayerList, Module
from max.pipelines.architectures.deepseekV3_2.model import DeepseekV3_2Model
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.vision_batching import create_empty_image_embeddings
from max.pipelines.lib.vision_encoder_cache import (
    VisionEncodeResult,
    concat_device_buffers,
)

from .batch_processor import Glm5NextBatchProcessor
from .context import Glm5NextTextAndVisionContext
from .glm5_next import Glm5Next
from .model_config import SPARSE_ATTENTION, Glm5NextConfig
from .nn.vision_geometry import vision_cu_seqlens, vision_position_ids
from .nn.visual_transformer import Glm5NextVisionTransformer
from .weight_adapters import VISION_WEIGHT_PREFIX

logger = logging.getLogger("max.pipelines")

__all__ = ["Glm5NextModel"]

# The decoder and the vision tower compile into one model.
_COMPILE_GROUP = "glm5_next"


def _mla_cache_dtype(config: Glm5NextConfig) -> DType:
    """Storage dtype of the latent cache.

    ``kv_params`` is a :class:`MultiKVCacheParams` tree holding the MLA latent
    and the indexer key cache separately; only the leaves carry a dtype.
    """
    params = config.kv_params
    leaf = (
        params.children["mla"]
        if isinstance(params, MultiKVCacheParams)
        else params
    )
    assert isinstance(leaf, KVCacheParams), (
        "The MLA latent cache must be a leaf KVCacheParams, got "
        f"{type(leaf).__name__}."
    )
    return leaf.dtype


class _VisionTowers(Module):
    """The vision tower, its output rows returned once per device.

    Either every device runs a replica on the same patches, or device 0 runs
    the only tower and copies its rows to the others, as deepseekV41 does. A
    replica costs about 1.1 GB of weights and repeats the encode's FLOPs, and
    the patches it reads cross the interconnect as well: 18.8 KB of float32
    per merged token, against 8.2 KB of bfloat16 rows. The encoder cache
    stores one copy of each row per device either way.
    """

    def __init__(self, config: Glm5NextConfig, *, replicate: bool) -> None:
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        self.devices = config.devices
        self.towers = LayerList(
            [
                Glm5NextVisionTransformer(
                    vision_config, device, use_subgraphs=config.use_subgraphs
                )
                for device in (
                    config.devices if replicate else config.devices[:1]
                )
            ]
        )

    def __call__(
        self,
        pixel_values: TensorValue,
        rot_pos_ids: TensorValue,
        row_offsets: TensorValue,
        max_seqlen: TensorValue,
    ) -> tuple[TensorValue, ...]:
        if len(self.towers) == 1:
            (tower,) = self.towers
            rows = tower(pixel_values, rot_pos_ids, row_offsets, max_seqlen)
            return tuple(rows.to(device) for device in self.devices)
        return tuple(
            tower(
                pixel_values.to(device),
                rot_pos_ids.to(device),
                row_offsets.to(device),
                max_seqlen,
            )
            for tower, device in zip(self.towers, self.devices, strict=True)
        )


class Glm5NextModel(DeepseekV3_2Model):
    """GLM-5.3-Flash pipeline model.

    Encodes images through
    :class:`~max.pipelines.lib.vision_encoder_cache.SupportsVisionEncoding`:
    the pipeline owns the encoder cache and hands the language graph the
    assembled rows and their scatter indices.
    """

    model_config_cls: ClassVar[type[Any]] = Glm5NextConfig
    batch_processor_cls: ClassVar[type[Glm5NextBatchProcessor]] = (
        Glm5NextBatchProcessor
    )

    _vision_weights_dict: dict[str, WeightData]
    _empty_vision_embeddings: list[Buffer] | None = None

    #: Whether every device runs its own vision tower rather than device 0
    #: encoding and copying the rows. Replicated until encode time and memory
    #: are measured both ways. The memory planner charges a tower and its
    #: activations per device either way: device 0 holds both regardless,
    #: and it bounds the evenly split KV cache.
    replicate_vision_tower: ClassVar[bool] = True

    @property
    def vision_model(self) -> Model | None:
        """The compiled vision tower, or ``None`` for a checkpoint without
        one."""
        return self._models.get(GraphRole.VISION)

    # Without allreduce, the decoder splits tokens across the TP ranks before
    # the expert dispatch (`token_parallel.py`), the layout the 2x MI355
    # recipe serves. Exporters pick the matching EP backend from it.
    ep_dispatch_token_layout: ClassVar[str] = "tp_sharded"

    def _first_mla_layer(self) -> int:
        """Reads the first sparse-MLA layer off the schedule.

        GLM-5.3-Flash interleaves KDA and sparse MLA on a period-4 schedule, so
        layer 0 is KDA and carries no ``kv_a_layernorm``. Derived rather than
        hardcoded to 3: the index follows ``layer_types``, and a sibling that
        retunes the period would otherwise probe the wrong layer and raise.
        """
        text_config = Glm5NextConfig._get_text_config(self.huggingface_config)
        for index, layer_type in enumerate(
            Glm5NextConfig.resolve_layer_types(text_config)
        ):
            if layer_type == SPARSE_ATTENTION:
                return index
        raise ValueError(
            "GLM-5.3-Flash needs at least one sparse-attention layer, but "
            "layer_types declares none."
        )

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> Glm5NextConfig:
        """Builds the config, then logs the two pools' per-sequence costs.

        Both pools are sized independently and either can cap concurrency
        first, so the crossover is worth having in the log of any run whose
        batch size later turns out to be the constraint.
        """
        config = super()._create_model_config(state_dict)
        assert isinstance(config, Glm5NextConfig)

        # The base class installs the shared blockscaled-FP8 map, which
        # describes every layer as quantized. Replace it: GLM-5.3-Flash leaves
        # the 34 KDA layers, the whole indexer and `kv_b_proj` in bfloat16, and
        # reading those as FP8 produces plausible tensors and garbage output.
        config.quant_config = config.resolve_quant_scheme(state_dict)

        self._glm_config = config
        state_bytes = config.per_request_state_bytes()
        logger.info(
            "GLM-5.3-Flash pools: %d KDA layers holding %.1f MiB per sequence "
            "at %s, context-independent; %d sparse-MLA layers holding %.1f KiB "
            "per token. The state pool binds below roughly 12K context and the "
            "KV cache above it.",
            len(config.kda_layers),
            state_bytes / 1024**2,
            config.state_dtype,
            len(config.sparse_attention_layers),
            len(config.sparse_attention_layers)
            * config.mla_head_dim
            * _mla_cache_dtype(config).size_in_bytes
            / 1024,
        )
        return config

    #: The resolved config, captured in :meth:`_create_model_config`. The base
    #: class keeps it as a local, and the state pools are allocated after the
    #: graph is built, so it has to be held somewhere.
    _glm_config: Glm5NextConfig | None = None

    def _load_state_dict(self) -> dict[str, Any]:
        """Loads the weights and sets the vision tower's aside, without their
        prefix, so the decoder's strict load sees only its own."""
        state_dict = super()._load_state_dict()
        self._vision_weights_dict = {
            name.removeprefix(VISION_WEIGHT_PREFIX): state_dict.pop(name)
            for name in [
                n for n in state_dict if n.startswith(VISION_WEIGHT_PREFIX)
            ]
        }
        return state_dict

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the GLM-5.3-Flash language graph and, for a checkpoint with
        a vision tower, the tower's graph, compiled together.

        The language graph's inputs are in the order
        :meth:`Glm5Next.input_types` declares them, which is the one place that
        order is written down. It stays :attr:`model`, the graph device-graph
        capture records; vision runs only on prefill steps.
        """
        assert isinstance(model_config, Glm5NextConfig)
        nn_model = Glm5Next(model_config, kv_params=self.kv_params)
        nn_model.attach_layers(nn_model.build_layers())
        nn_model.load_state_dict(
            self._weights_dict, weight_alignment=1, strict=True
        )
        specs: dict[str, GraphSpec] = {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "glm5_next_graph",
                nn_model.input_types(self.kv_params),
                weights=nn_model.state_dict(),
                group=_COMPILE_GROUP,
            )
        }
        if model_config.vision_config is not None:
            specs[GraphRole.VISION] = self._vision_spec(
                model_config, self._vision_weights_dict
            )
        return specs

    def _vision_spec(
        self, config: Glm5NextConfig, state_dict: dict[str, WeightData]
    ) -> GraphSpec:
        """Returns the vision graph: its towers, built from shared host
        weights."""
        vision_config = config.vision_config
        assert vision_config is not None
        towers = _VisionTowers(config, replicate=self.replicate_vision_tower)
        # Keyed by replica index; the host arrays are shared, not copied.
        towers.load_state_dict(
            {
                f"towers.{replica}.{name}": weight
                for replica in range(len(towers.towers))
                for name, weight in state_dict.items()
            },
            weight_alignment=1,
            strict=True,
        )
        patch_dim = (
            vision_config.in_channels
            * vision_config.temporal_patch_size
            * vision_config.patch_size**2
        )
        device0 = config.devices[0]
        return GraphSpec(
            towers,
            "glm5_next_vision_graph",
            [
                TensorType(
                    DType.float32, ["vision_patches", patch_dim], device0
                ),
                TensorType(DType.int32, ["vision_patches", 2], device0),
                TensorType(DType.uint32, ["vision_segments"], device0),
                TensorType(DType.uint32, [1], DeviceRef.CPU()),
            ],
            weights=towers.state_dict(),
            group=_COMPILE_GROUP,
        )

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Glm5NextTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> None:
        """Packs inline in :meth:`vision_execute` instead."""
        return

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Glm5NextTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: None,
    ) -> VisionEncodeResult:
        """Encodes the cache-miss images, context-major then image-minor.

        Runs in chunks of at most ``max_batch_input_tokens`` output rows, the
        same bound the decoder honors, so the tower's activations stay within
        what the memory planner reserved. An image is never split.
        """
        assert self.vision_model is not None, (
            "This checkpoint has no vision tower, so it cannot encode images."
        )
        config = self._glm_config
        assert config is not None and config.vision_config is not None
        merge_unit = config.vision_config.spatial_merge_size**2

        entries: list[
            tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]
        ] = []
        for ctx, miss_images in selection:
            # Spans never overlap, so a start index names one image even when
            # the miss list holds a reconstructed `ImageMetadata`.
            index_of = {img.start_idx: i for i, img in enumerate(ctx.images)}
            for img in miss_images:
                i = index_of[img.start_idx]
                entries.append((ctx.images[i].pixel_values, ctx.grid_thws[i]))
        counts = [int(np.prod(grid)) // merge_unit for _, grid in entries]

        budget = self.pipeline_config.runtime.max_batch_input_tokens
        chunks: list[list[int]] = [[]]
        chunk_rows = 0
        for index, count in enumerate(counts):
            if chunks[-1] and chunk_rows + count > budget:
                chunks.append([])
                chunk_rows = 0
            chunks[-1].append(index)
            chunk_rows += count

        outputs = [
            self._encode_images([entries[i] for i in chunk]) for chunk in chunks
        ]
        embeddings = (
            outputs[0]
            if len(outputs) == 1
            else [
                concat_device_buffers([output[d] for output in outputs])
                for d in range(len(devices))
            ]
        )
        return VisionEncodeResult(
            embeddings=embeddings, per_image_token_counts=counts
        )

    def _encode_images(
        self,
        entries: Sequence[
            tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]
        ],
    ) -> list[Buffer]:
        """Runs the tower once; returns one ``[rows, hidden]`` copy per device."""
        assert self.vision_model is not None
        config = self._glm_config
        assert config is not None and config.vision_config is not None
        device0 = self.devices[0]
        grids = np.stack([grid for _, grid in entries])
        row_offsets = vision_cu_seqlens(grids)
        max_seqlen = np.array(
            [np.diff(row_offsets.astype(np.int64)).max()], dtype=np.uint32
        )
        # Packed straight into staging memory: a concatenated numpy copy is a
        # fresh allocation of up to hundreds of MB per chunk, and on a host
        # with a fragmented page cache its huge-page faults compact
        # synchronously.
        pixels = Buffer(
            dtype=DType.float32,
            shape=(
                sum(len(entry) for entry, _ in entries),
                entries[0][0].shape[1],
            ),
            device=device0,
            usage=Usage.DEFAULT
            if device0.is_host
            else Usage.STAGING | Usage.UNTRACKED,
        )
        staged = pixels.to_numpy()
        row = 0
        for entry, _ in entries:
            staged[row : row + len(entry)] = entry
            row += len(entry)
        outputs = self.vision_model.execute(
            pixels.to(device0),
            Buffer.from_numpy(
                vision_position_ids(
                    grids, config.vision_config.spatial_merge_size
                )
            ).to(device0),
            Buffer.from_numpy(row_offsets).to(device0),
            Buffer.from_numpy(max_seqlen),
        )
        assert len(outputs) == len(self.devices)
        buffers: list[Buffer] = []
        for output in outputs:
            assert isinstance(output, Buffer)
            buffers.append(output)
        return buffers

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device ``[0, hidden]`` rows for a step that encodes nothing.

        Cached, because every decode step asks for them.
        """
        if self._empty_vision_embeddings is None:
            text_config = Glm5NextConfig._get_text_config(
                self.huggingface_config
            )
            self._empty_vision_embeddings = create_empty_image_embeddings(
                devices, text_config.hidden_size
            )
        return self._empty_vision_embeddings
