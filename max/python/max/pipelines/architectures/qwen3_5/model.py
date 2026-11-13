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
"""The Qwen3.5 pipeline model."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from max import tree
from max._artifact.vision import (
    ModalityLayout,
    PerClip,
    PromptText,
    SmartResize,
    Span,
    TemplateMarkers,
    VideoSmartResize,
    VisionMetadata,
)
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.nn.kv_cache import KVCacheParamInterface, recurrent_leaf
from max.nn.legacy.image_preprocess import (
    image_preprocess_input_types,
    video_preprocess_input_types,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    WeightEntry,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from transformers import AutoConfig, PretrainedConfig

from ..llama3.model import Llama3Inputs, LlamaModelBase
from ..qwen3vl_moe.context import Qwen3VLTextAndVisionContext
from ..qwen3vl_moe.nn.data_processing import QWEN3VL_MAX_PIXELS
from ..qwen3vl_moe.nn.video_processing import (
    QWEN3VL_VIDEO_MAX_PIXELS,
    QWEN3VL_VIDEO_MIN_PIXELS,
)
from ..qwen3vl_moe.tokenizer import VISION_END_TOKEN, VISION_START_TOKEN
from .batch_processor import Qwen3_5BatchProcessor
from .model_config import Qwen3_5Config
from .preprocess_graph import Qwen3_5ImagePreprocess
from .qwen3_5 import Qwen3_5
from .vision_packing import Qwen3_5VisionInputs
from .weight_adapters import qwen3_5_weight_entry

logger = logging.getLogger("max.pipelines")

# The language and vision graphs compile into one artifact.
_COMPILE_GROUP = "qwen3_5"


@dataclass
class Qwen3_5Inputs(Llama3Inputs):
    """Inputs for Qwen3.5, including the linear-attention state pools.

    Image embeddings come from the pipeline-driven encoder cache on the base
    ``vision_embeddings`` / ``vision_scatter_indices`` fields.
    """

    request_ids: list[RequestID] | None = None
    """Request IDs for this batch, used to manage per-request state cache slots."""

    decoder_position_ids: Buffer | None = None
    """``[3, total_seq_len]`` M-RoPE positions, one column per active token.

    Present only when the graph was built with M-RoPE wired in; see
    ``Qwen3_5.mrope_enabled``."""

    vision_inputs_in_graph: bool = True
    """Whether the graph declares the vision-merge inputs. False for
    text-only checkpoints, which compile no vision encoder and so take none."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, in the graph's input order."""
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
            # Set by the pipeline's vision seam (``finalize_vision_inputs``)
            # on every prepared batch, empties included.
            *(self.vision_embeddings if self.vision_inputs_in_graph else ()),
            *(
                self.vision_scatter_indices
                if self.vision_inputs_in_graph
                else ()
            ),
            *(
                ()
                if self.decoder_position_ids is None
                else (self.decoder_position_ids,)
            ),
        )


# Scale tensors carry the calibration a quantized checkpoint cannot be read
# without. A dropped one is not a degradation, it is a different model.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# Everything the architecture deliberately does not load. `mtp.*` is the
# speculative-decoding head, which the weight adapter drops.
_UNUSED_PREFIXES = ("mtp.",)


def _check_weights_match(expected: set[str], provided: set[str]) -> None:
    """Fails the load when the checkpoint and the graph disagree on weights.

    ``load_state_dict(strict=False)`` drops both directions of mismatch
    without a word, so a quantized checkpoint whose 995 scale tensors go
    unconsumed loads clean and emits garbage. This is the gate that turns
    that into a startup error.

    Args:
        expected: Weight names the built graph will look up.
        provided: Weight names the adapted checkpoint supplies.

    Raises:
        ValueError: If a weight the graph needs is absent, or a scale tensor
            the checkpoint supplies is not consumed.
    """
    missing = sorted(expected - provided)
    if missing:
        raise ValueError(
            f"Qwen3.5 checkpoint is missing {len(missing)} weight(s) the model "
            f"requires: {missing[:20]}"
            + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
        )

    unused = provided - expected
    unconsumed_scales = sorted(
        k
        for k in unused
        if k.endswith(_SCALE_SUFFIXES) and not k.startswith(_UNUSED_PREFIXES)
    )
    if unconsumed_scales:
        raise ValueError(
            f"Qwen3.5 checkpoint supplies {len(unconsumed_scales)} "
            "quantization scale tensor(s) that no layer consumes, so those "
            "weights would be read at the wrong precision: "
            f"{unconsumed_scales[:20]}"
            + (
                f" (+{len(unconsumed_scales) - 20} more)"
                if len(unconsumed_scales) > 20
                else ""
            )
        )

    if unused:
        logger.info(
            "Qwen3.5 load_state_dict: %d unused checkpoint keys: %s",
            len(unused),
            sorted(unused)[:20],
        )


def _loaded_module(
    config: Qwen3_5Config, state_dict: dict[str, Any]
) -> Qwen3_5:
    module = Qwen3_5(config)
    _check_weights_match(
        expected=set(module.raw_state_dict().keys()),
        provided=set(state_dict.keys()),
    )
    module.load_state_dict(
        state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=False,
    )
    return module


# The model names its tower's weights under this prefix, as the checkpoint's
# rules do.
_VISION_PREFIX = "vision_encoder."


def _vision_spec(model: Qwen3_5) -> GraphSpec | None:
    """Returns the graph of ``model``'s vision tower, if it has one."""
    if model.vision_encoder is None:
        return None
    return GraphSpec(
        model.vision_encoder,
        "qwen3_5_vision",
        model.vision_encoder.input_tree(),
        weights={
            name: value
            for name, value in model.state_dict().items()
            if name.startswith(_VISION_PREFIX)
        },
        group=_COMPILE_GROUP,
    )


def qwen3_5_vision_metadata(
    huggingface_config: AutoConfig,
) -> VisionMetadata | None:
    """Returns the media layout a Qwen3.5 checkpoint is served with.

    Returns ``None`` for a text-only checkpoint. The chat template writes the vision start and end markers around each
    placeholder as text, so the spliced runs are bare placeholders.
    """
    vision = getattr(huggingface_config, "vision_config", None)
    if vision is None:
        return None
    markers = TemplateMarkers(
        huggingface_config.vision_start_token_id,
        huggingface_config.vision_end_token_id,
    )
    return VisionMetadata.from_layouts(
        image=ModalityLayout(
            patch_size=vision.patch_size,
            merge_size=vision.spatial_merge_size,
            # The min_pixels Qwen3VLTokenizer gives its image processor.
            strategy=SmartResize(
                min_pixels=65536, max_pixels=QWEN3VL_MAX_PIXELS
            ),
            span=Span(huggingface_config.image_token_id),
            template_markers=markers,
        ),
        video=ModalityLayout(
            patch_size=vision.patch_size,
            merge_size=vision.spatial_merge_size,
            strategy=VideoSmartResize(
                temporal_patch_size=vision.temporal_patch_size,
                min_pixels=QWEN3VL_VIDEO_MIN_PIXELS,
                max_pixels=QWEN3VL_VIDEO_MAX_PIXELS,
            ),
            span=Span(huggingface_config.video_token_id),
            template_markers=markers,
            # The clip budget bounds every frame at once, so a clip's frame
            # count changes how each frame is resized.
            video_layout=PerClip(),
        ),
        # MAX declares no byte caps for this family; 512 MiB is the serving
        # engine's own video cap, which real lecture clips need.
        max_image_bytes=0,
        max_video_bytes=512 << 20,
        prompt=PromptText(
            video_clip_open=VISION_START_TOKEN,
            video_clip_close=VISION_END_TOKEN,
        ),
    )


def _preprocess(vision: PretrainedConfig) -> Qwen3_5ImagePreprocess:
    """Returns the preprocessing module for a checkpoint's vision config."""
    return Qwen3_5ImagePreprocess(
        patch_size=vision.patch_size,
        temporal_patch_size=vision.temporal_patch_size,
        merge_size=vision.spatial_merge_size,
        num_position_embeddings=vision.num_position_embeddings,
    )


class Qwen3_5Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Qwen3.5 pipeline model implementation.

    Supports the hybrid linear/full attention architecture with KV cache
    for full attention layers and conv/recurrent states for linear layers.
    """

    model_config_cls: ClassVar[type[Any]] = Qwen3_5Config
    batch_processor_cls: ClassVar[type[Qwen3_5BatchProcessor]] = (
        Qwen3_5BatchProcessor
    )

    model: Model
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    _models: Mapping[str, Model]
    # The adapted checkpoint, which `graph_specs` loads into the modules.
    _checkpoint: dict[str, Any]

    # State-pool storage dtype, which can be wider than the compute dtype.
    _state_dtype: DType = DType.bfloat16
    # Per-request state bytes the config budgeted, checked in `load_model`.
    _accounted_state_bytes: int = 0

    _num_linear_layers: int = 0

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Qwen3_5, Qwen3_5Config]:
        """Builds the language model from the checkpoint ``context`` names.

        A caller that sized the cache passes its KV params. The vision tower
        builds in its own graph, as :meth:`build_vision_spec` returns it, so
        the module keeps only the 3-axis positions of a checkpoint that has
        one.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        config = cls._finalized_config(context, state_dict, vision=False)
        return _loaded_module(config, state_dict), config

    @classmethod
    def build_vision_spec(cls, context: ModelLoadContext) -> GraphSpec:
        """Returns the vision tower's graph spec.

        Raises:
            ValueError: If the checkpoint has no vision tower, or no
                ``mrope_section`` to place the image tokens.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        config = cls._finalized_config(context, state_dict)
        spec = _vision_spec(_loaded_module(config, state_dict))
        if spec is None:
            raise ValueError("This Qwen3.5 checkpoint has no vision tower.")
        if not config.mrope_enabled:
            raise ValueError(
                "This Qwen3.5 checkpoint declares no mrope_section, so its"
                " image tokens would take flat positions."
            )
        return spec

    @classmethod
    def weight_entries(
        cls, context: ModelLoadContext, config: Qwen3_5Config
    ) -> list[WeightEntry]:
        """Returns the entry that names the built model's weights.

        Its rules name the whole checkpoint, and the model takes the weights
        it declares.
        """
        del config
        return [qwen3_5_weight_entry(context.adapter)]

    @classmethod
    def _finalized_config(
        cls,
        context: ModelLoadContext,
        state_dict: dict[str, Any],
        *,
        vision: bool = True,
    ) -> Qwen3_5Config:
        huggingface_config = context.huggingface_config
        config = Qwen3_5Config.initialize_from_config(
            context.pipeline_config,
            huggingface_config,
            max_seq_len=context.max_seq_len,
        )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParamInterface)
            config.kv_params = context.kv_params
        if not vision:
            # Tokens after an image still take its positions.
            config.mrope_without_encoder = config.vision_config is not None
            config.vision_config = None
        config.finalize(
            huggingface_config=Qwen3_5Config._get_text_config(
                huggingface_config
            ),
            state_dict=state_dict,
            return_logits=context.return_logits,
            return_hidden_states=context.return_hidden_states,
            norm_method=cls.norm_method,
            attention_bias=cls.attention_bias,
        )
        if context.target_layer_ids:
            config.target_layer_ids = list(context.target_layer_ids)

        # finalize() reads tie_word_embeddings from the text sub-config,
        # which inherits PretrainedConfig's default of True.  The correct
        # value lives on the top-level config.
        config.tie_word_embeddings = getattr(
            huggingface_config, "tie_word_embeddings", False
        )
        return config

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        """Loads the graphs into ``session`` and returns the decoder."""
        self._input_row_offsets_prealloc: Buffer | None = None
        max_batch_size = self.max_batch_size
        assert max_batch_size is not None, (
            "max_batch_size must be set in runtime config"
        )
        if not is_virtual_device_mode():
            self._input_row_offsets_prealloc = Buffer.from_numpy(
                np.arange(
                    max_batch_size + 1,
                    dtype=np.uint32,
                )
            ).to(self.devices[0])

        self._models = self.load_graphs(session)
        vision_model = self._models.get(GraphRole.VISION)
        if vision_model is not None:
            self._batching.bind_vision_encoder(
                vision_model=vision_model, session=session
            )
        # The modules hold what they need of it; keeping the checkpoint would
        # pin its host copies for the model's lifetime.
        self._checkpoint = {}

        if self._num_linear_layers > 0 and not is_virtual_device_mode():
            self.check_state_budget()

        return self._models[GraphRole.DECODER]

    def check_state_budget(self) -> None:
        """Checks the state the cache declares against what planning budgeted.

        The two resolve ``state_dtype`` either side of ``finalize``, and a
        disagreement surfaces only as an OOM at load.
        """
        state = recurrent_leaf(self.kv_params)
        assert state is not None, (
            "Qwen3.5 has linear-attention layers, so its cache must declare a"
            " recurrent state child"
        )
        allocated = state.bytes_per_state * len(self.devices)
        assert allocated == self._accounted_state_bytes, (
            f"Qwen3.5 declares {allocated} B of recurrent state per request"
            f" but memory planning budgeted {self._accounted_state_bytes} B."
            f" The pool dtype ({self._state_dtype}) and the accounted dtype"
            " must agree."
        )

    @property
    def _batching(self) -> Qwen3_5BatchProcessor:
        assert isinstance(self._batch_processor, Qwen3_5BatchProcessor)
        return self._batch_processor

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> Qwen3_5VisionInputs | None:
        """Packs the selected images, as the batch processor does."""
        return self._batching.pack_vision_inputs(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: Qwen3_5VisionInputs | None,
    ) -> VisionEncodeResult:
        """Encodes the selected images, as the batch processor does."""
        return self._batching.vision_execute(selection, devices, packed)

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns the batch processor's embeddings for a batch with no new images."""
        return self._batching.empty_vision_embeddings(devices)

    def _load_state_dict(self) -> dict[str, Any]:
        self._checkpoint = super()._load_state_dict()
        # One entry names the whole checkpoint; each graph binds its part.
        entry = qwen3_5_weight_entry(self.adapter)
        self.weight_naming = {
            GraphRole.DECODER: (entry,),
            GraphRole.VISION: (entry,),
        }
        return self._checkpoint

    def _create_model_config(self, state_dict: dict[str, Any]) -> Qwen3_5Config:
        return self._finalized_config(
            ModelLoadContext.from_pipeline_model(self), state_dict
        )

    @classmethod
    def vision_metadata_for(
        cls, pipeline_config: PipelineConfig, config: object
    ) -> VisionMetadata | None:
        """Returns the checkpoint's vision metadata."""
        del config
        huggingface_config = pipeline_config.model.huggingface_config
        assert huggingface_config is not None
        return qwen3_5_vision_metadata(huggingface_config)

    @classmethod
    def vision_preprocess_spec_for(
        cls, pipeline_config: PipelineConfig, config: object
    ) -> GraphSpec | None:
        """Returns the graph that preprocesses one image for the tower.

        It does what the tokenizer and the vision input packing do.
        """
        del config
        huggingface_config = pipeline_config.model.huggingface_config
        vision = getattr(huggingface_config, "vision_config", None)
        if vision is None:
            return None
        return GraphSpec(
            _preprocess(vision),
            "qwen3_5_vision_preprocess",
            image_preprocess_input_types(),
            weights={},
        )

    @classmethod
    def vision_preprocess_video_spec_for(
        cls, pipeline_config: PipelineConfig, config: object
    ) -> GraphSpec | None:
        """Returns the graph that preprocesses one temporal patch of a clip.

        It does for the tower what the video processor does. The tower
        attends within each temporal patch, so a patch is a clip's attention
        segment.
        """
        del config
        huggingface_config = pipeline_config.model.huggingface_config
        vision = getattr(huggingface_config, "vision_config", None)
        if vision is None:
            return None
        return GraphSpec(
            _preprocess(vision),
            "qwen3_5_vision_preprocess_video",
            video_preprocess_input_types(vision.temporal_patch_size, 1),
            weights={},
        )

    def graph_specs(self, model_config: Qwen3_5Config) -> dict[str, GraphSpec]:
        """Returns the decoder and the vision encoder as one compile group.

        A text-only checkpoint has no vision encoder, so the group holds the
        decoder alone.
        """
        nn_model = _loaded_module(model_config, self._checkpoint)
        input_types = self._bind_graph_inputs(
            nn_model.input_tree(self.kv_params)
        )

        # The decoder takes the language model's weights, and the vision
        # tower's spec its own.
        self.state_dict = {
            k: v
            for k, v in nn_model.state_dict().items()
            if not k.startswith(_VISION_PREFIX)
        }

        self._num_linear_layers = len(nn_model.linear_layer_indices)
        self._state_dtype = model_config.state_dtype
        self._accounted_state_bytes = model_config._per_request_state_bytes()
        self._batching.compute_dtype = model_config.compute_dtype

        specs: dict[str, GraphSpec] = {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "qwen3_5",
                input_types,
                weights=self.state_dict,
                group=_COMPILE_GROUP,
            )
        }
        # Text-only checkpoints have no vision tower, hence no vision graph.
        if (vision := _vision_spec(nn_model)) is not None:
            specs[GraphRole.VISION] = vision
        return specs

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs one decoding step and returns its logits."""
        assert isinstance(model_inputs, Qwen3_5Inputs)
        assert model_inputs.kv_cache_inputs is not None

        model_outputs = self.model.execute(*model_inputs.buffers)

        # The slot-indexed SSM kernels mutate the conv/recurrent pools in
        # place; the only graph output is the logits.
        logits = model_outputs[0]
        assert isinstance(logits, Buffer)

        return ModelOutputs(
            logits=logits,
            next_token_logits=logits,
        )
