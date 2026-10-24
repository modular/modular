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

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from max import tree
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.buffer_utils import cast_tensors_to
from max.nn.kv_cache import KVCacheParamInterface, recurrent_leaf
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.pipelines.modeling.types import RequestID
from max.profiler import traced

from ..llama3.model import Llama3Inputs, LlamaModelBase
from ..qwen3vl_moe.context import Qwen3VLTextAndVisionContext
from .batch_processor import Qwen3_5BatchProcessor
from .model_config import Qwen3_5Config
from .qwen3_5 import Qwen3_5
from .vision_packing import Qwen3_5VisionInputs, pack_uncached_images

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

    # Vision model (None for text-only checkpoints)
    vision_model: Model | None = None
    _models: Mapping[str, Model]
    # The adapted checkpoint, which `graph_specs` loads into the modules.
    _checkpoint: dict[str, Any]
    _session: InferenceSession | None = None

    # Model dtype and hidden size (set during graph build, used for empty buffers)
    _hidden_size: int = 0
    _model_dtype: DType = DType.bfloat16
    # State-pool storage dtype. Deliberately separate from _model_dtype: the
    # vision empties keep the compute dtype even when the pools are fp32.
    _state_dtype: DType = DType.bfloat16
    # Per-request state bytes the config budgeted, checked in `load_model`.
    _accounted_state_bytes: int = 0

    _num_linear_layers: int = 0

    # Whether the built graph takes M-RoPE positions; set during graph build.
    _mrope_enabled: bool = False

    # Zero-row vision embeddings for decode / text-only steps, so buffers()
    # always has the right input count. Cached: see empty_vision_embeddings.
    _empty_vision_embeddings: list[Buffer] | None = None

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Qwen3_5, Qwen3_5Config]:
        """Builds the hybrid model from the checkpoint ``context`` names.

        A caller that sized the cache passes its KV params. A module that
        encodes no images builds no vision tower, but keeps the 3-axis
        positions of a checkpoint that has one.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        config = cls._finalized_config(context, state_dict)
        return _loaded_module(config, state_dict), config

    @classmethod
    def _finalized_config(
        cls, context: ModelLoadContext, state_dict: dict[str, Any]
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
        if not context.encodes_images:
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
        self._session = session

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
        self.vision_model = self._models.get(GraphRole.VISION)
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

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> Qwen3_5VisionInputs | None:
        """Pack the pipeline-selected cache-miss images to device.

        Runs in the pipeline's prep-ahead window so the host-to-device copy
        overlaps the prior batch.
        """
        return pack_uncached_images(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[
            tuple[Qwen3VLTextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: Qwen3_5VisionInputs | None,
    ) -> VisionEncodeResult:
        """Run the vision encoder on the media ``pack_vision_inputs`` packed.

        Returns explicit per-item token counts rather than letting the driver
        derive them from placeholder spans. A clip's span covers ``grid_t``
        placeholder runs AND the timestamp text between them, so its width
        exceeds the rows the encoder emits; an image's span happens to match,
        which is why the derived counts were correct until now.
        """
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(devices)
            )
        assert self.vision_model is not None
        assert self._session is not None
        vision_outputs = self.vision_model.execute(
            packed.pixel_values,
            packed.weights,
            packed.indices,
            packed.vision_position_ids,
            packed.max_grid_size,
            packed.grid_thw,
            packed.cu_seqlens,
            packed.max_seqlen,
            *self.signal_buffers,
        )
        assert isinstance(vision_outputs[0], Buffer)
        embeddings = cast_tensors_to(
            [vision_outputs[0]], self._model_dtype, self._session
        )[0]
        # The hidden state is replicated across devices, so every replica
        # merges the same embeddings.
        return VisionEncodeResult(
            embeddings=[embeddings.to(device) for device in devices],
            per_image_token_counts=[
                img.embedding_rows
                for _ctx, miss_images in selection
                for img in miss_images
            ],
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: hit on every text-only / decode step, so it must not allocate
        per call, and graph-capture replay only skips an input refresh for an
        identical buffer object.
        """
        if self._empty_vision_embeddings is None:
            self._empty_vision_embeddings = [
                Buffer.zeros(
                    shape=[0, self._hidden_size], dtype=self._model_dtype
                ).to(device)
                for device in devices
            ]
        return self._empty_vision_embeddings

    def _load_state_dict(self) -> dict[str, Any]:
        self._checkpoint = super()._load_state_dict()
        return self._checkpoint

    def _create_model_config(self, state_dict: dict[str, Any]) -> Qwen3_5Config:
        return self._finalized_config(
            ModelLoadContext.from_pipeline_model(self), state_dict
        )

    def graph_specs(self, model_config: Qwen3_5Config) -> dict[str, GraphSpec]:
        """Returns the decoder, and the vision encoder unless the checkpoint
        is text-only, in one compile group."""
        nn_model = _loaded_module(model_config, self._checkpoint)
        input_types = self._bind_graph_inputs(
            nn_model.input_tree(self.kv_params)
        )

        # Split the processed state dict into vision and LM parts. Vision
        # keys keep their "vision_encoder." prefix: the weights are named
        # from nn_model, the root, when it loads them.
        processed = nn_model.state_dict()
        vision_prefix = "vision_encoder."
        vision_state_dict = {
            k: v for k, v in processed.items() if k.startswith(vision_prefix)
        }
        self.state_dict = {
            k: v
            for k, v in processed.items()
            if not k.startswith(vision_prefix)
        }

        self._num_linear_layers = len(nn_model.linear_layer_indices)
        self._hidden_size = model_config.hidden_size
        self._model_dtype = model_config.compute_dtype
        self._state_dtype = model_config.state_dtype
        self._accounted_state_bytes = model_config._per_request_state_bytes()

        vision_encoder = nn_model.vision_encoder
        self._mrope_enabled = nn_model.mrope_enabled
        if isinstance(self._batch_processor, Qwen3_5BatchProcessor):
            self._batch_processor.mrope_enabled = self._mrope_enabled
            self._batch_processor.vision_enabled = vision_encoder is not None

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
        if vision_encoder is not None:
            specs[GraphRole.VISION] = GraphSpec(
                vision_encoder,
                "qwen3_5_vision",
                vision_encoder.input_types(),
                weights=vision_state_dict,
                group=_COMPILE_GROUP,
            )
        return specs

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
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
