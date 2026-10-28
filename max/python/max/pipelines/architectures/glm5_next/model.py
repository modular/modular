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
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache.cache_params import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV3_2.model import DeepseekV3_2Model
from max.pipelines.architectures.deepseekV3_2.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from transformers import AutoConfig
from typing_extensions import override

from .batch_processor import Glm5NextBatchProcessor
from .context import Glm5NextTextAndVisionContext
from .glm5_next import Glm5Next
from .model_config import SPARSE_ATTENTION, Glm5NextConfig
from .nn.visual_transformer import Glm5NextVisionTransformer
from .weight_adapters import VISION_CHECKPOINT_PREFIX, VISION_WEIGHT_PREFIX

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

    @classmethod
    def _first_mla_layer(cls, huggingface_config: AutoConfig) -> int:
        """Reads the first sparse-MLA layer off the schedule.

        GLM-5.3-Flash interleaves KDA and sparse MLA on a period-4 schedule, so
        layer 0 is KDA and carries no ``kv_a_layernorm``. Derived rather than
        hardcoded to 3: the index follows ``layer_types``, and a sibling that
        retunes the period would otherwise probe the wrong layer and raise.
        """
        text_config = Glm5NextConfig._get_text_config(huggingface_config)
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

    @override
    @classmethod
    def finalized_config(
        cls,
        pipeline_config: PipelineConfig,
        state_dict: Mapping[str, Any],
        *,
        max_seq_len: int,
        max_batch_total_tokens: int | None,
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates,
        model_config: DeepseekV3_2Config | None = None,
    ) -> Glm5NextConfig:
        """Builds the config, with the precision map read off ``state_dict``."""
        config = super().finalized_config(
            pipeline_config,
            state_dict,
            max_seq_len=max_seq_len,
            max_batch_total_tokens=max_batch_total_tokens,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            model_config=model_config,
        )
        assert isinstance(config, Glm5NextConfig)
        # The base class installs the shared blockscaled-FP8 map, which
        # describes every layer as quantized. Replace it: GLM-5.3-Flash leaves
        # the 34 KDA layers, the whole indexer and `kv_b_proj` in bfloat16, and
        # reading those as FP8 produces plausible tensors and garbage output.
        config.quant_config = config.resolve_quant_scheme(state_dict)
        return config

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Glm5Next, Glm5NextConfig]:
        """Builds the decoder from the checkpoint ``context`` names, without
        the vision tower, which :meth:`build_vision_graph` builds.

        A caller that sized the cache passes its KV params. It starts expert
        parallelism's runtime through ``context`` before it shards the
        experts.
        """
        state_dict = {
            name: weight
            for name, weight in parse_state_dict_from_weights(
                context.pipeline_config, context.weights, context.adapter
            ).items()
            if not name.startswith(VISION_WEIGHT_PREFIX)
        }
        config = cls.finalized_config(
            context.pipeline_config,
            state_dict,
            max_seq_len=context.max_seq_len,
            max_batch_total_tokens=context.max_batch_total_tokens,
            return_logits=context.return_logits,
            return_hidden_states=context.return_hidden_states,
        )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, MultiKVCacheParams)
            config.kv_params = context.kv_params
        context.start_ep(config.ep_config)
        module = Glm5Next(config, kv_params=config.kv_params)
        module.attach_layers(module.build_layers())
        module.load_state_dict(state_dict, weight_alignment=1, strict=True)
        return module, config

    @classmethod
    def build_vision_graph(
        cls, context: ModelLoadContext
    ) -> tuple[Graph, dict[str, Any]]:
        """Builds the vision tower's graph and the weights it loads, for a
        caller that runs the module :meth:`build_module` builds.

        Raises:
            ValueError: If the checkpoint has no vision tower.
        """
        config = Glm5NextConfig.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        if config.vision_config is None:
            raise ValueError(
                "This GLM-5.3-Flash checkpoint has no vision tower."
            )
        # The adapter converts each tower weight on its own, so it is handed
        # only the tower: a whole-checkpoint pass would concatenate and upcast
        # every text layer again just to drop it.
        assert context.adapter is not None
        state_dict = context.adapter(
            {
                name: weight
                for name, weight in context.weights.items()
                if name.startswith(VISION_CHECKPOINT_PREFIX)
            },
            huggingface_config=context.pipeline_config.model.huggingface_config,
            pipeline_config=context.pipeline_config,
        )
        spec = cls._vision_spec(
            config,
            {
                name.removeprefix(VISION_WEIGHT_PREFIX): weight
                for name, weight in state_dict.items()
            },
        )
        return spec.trace(), dict(spec.weights_registry())

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

    @classmethod
    def _vision_spec(
        cls, config: Glm5NextConfig, state_dict: dict[str, WeightData]
    ) -> GraphSpec:
        """Returns the vision graph: its towers, built from shared host
        weights."""
        vision_config = config.vision_config
        assert vision_config is not None
        towers = _VisionTowers(config, replicate=cls.replicate_vision_tower)
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

    @override
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Loads the graphs, then binds the vision tower to the batching."""
        models = super().load_graphs(session)
        if GraphRole.VISION in models:
            self._batching.bind_vision_encoder(
                vision_model=models[GraphRole.VISION], session=session
            )
        return models

    @property
    def _batching(self) -> Glm5NextBatchProcessor:
        assert isinstance(self._batch_processor, Glm5NextBatchProcessor)
        return self._batch_processor

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Glm5NextTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> None:
        """Packs as the batch processor does."""
        return self._batching.pack_vision_inputs(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Glm5NextTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: None,
    ) -> VisionEncodeResult:
        """Encodes the selected images, as the batch processor does."""
        return self._batching.vision_execute(selection, devices, packed)

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns the batch processor's rows for a step that encodes
        nothing."""
        return self._batching.empty_vision_embeddings(devices)
