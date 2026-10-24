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
"""Spec-decode pipeline models whose one graph runs the target and the draft.

:class:`SpecDecodePipelineModel` serves every pairing whose target and draft
are both registered architectures. Each builds its module with its own
pipeline model's
:meth:`~max.pipelines.lib.PipelineModel.build_module`, the
configured speculative method picks the proposer the draft runs through, and
a driver fuses the three into one graph. Neither pipeline model is
constructed, so nothing about how either compiles or executes reaches the
fused graph.

``_UnifiedSpecDecodeModelMixin`` is what the per-pairing unified packages
share, until each moves onto :class:`SpecDecodePipelineModel`.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from max import tree
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.graph.weights import (
    Weights,
    WeightsAdapter,
    load_weights,
    weights_format,
)
from max.nn.comm import Signals
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.kv_cache import (
    CacheLeafParamInterface,
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
)
from max.nn.sampling.penalties import LogitPenalties
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
)
from max.pipelines.kv_cache.memory_planner import MemoryPlanner
from max.pipelines.modeling.types import InputModality, RequestID
from max.pipelines.speculative import proposers, spec_adapters
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.config import (
    SpeculativeConfig,
    SpeculativeMethod,
)
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_adapters import STAGE_BOUNDARY
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.profiler import traced
from transformers import AutoConfig

from ..arch_lookup import ARCH_LOOKUP, SupportedArchitecture
from ..interfaces.arch_config import ArchConfigWithVisionCache
from ..interfaces.batch_processor import (
    BatchProcessorRuntime,
    UnifiedSpecDecodeBatchProcessor,
)
from ..interfaces.pipeline_model import (
    AlwaysSignalBuffersMixin,
    ModelInputs,
    ModelLoadContext,
    PipelineModel,
    PipelineModelWithKVCache,
    UnifiedEagleOutputs,
    UnifiedSpecDecodeInputs,
)
from ..memory_estimation import MemoryPlan
from ..pipeline_parallel import PipelinedModel
from ..pipeline_stage import PipelineStage
from ..utils import CompilationTimer
from ..vision_encoder_cache import SupportsVisionEncoding, VisionEncodeResult

if TYPE_CHECKING:
    from ..config import KVCacheConfig, MAXModelConfig, PipelineConfig

__all__ = [
    "SpecDecodeArchConfig",
    "SpecDecodePipelineModel",
    "SpecGraph",
    "_UnifiedSpecDecodeModelMixin",
    "build_spec_graph",
    "spec_decode_architecture",
]

logger = logging.getLogger("max.pipelines")


class _UnifiedSpecDecodeModelMixin:
    """Shared execute / _next_seed for unified spec-decode models."""

    # Provided by the concrete PipelineModel this mixin is combined with.
    model: Model
    devices: list[Device]

    # Dflash-style block drafters derive the draft width from the draft
    # checkpoint at model construction and expose it here instead of
    # rewriting SpeculativeConfig; None means the config already carries
    # the value.
    resolved_num_speculative_tokens: int | None = None

    @property
    def _spec_decode_model(self) -> Model:
        # self.model by default; Kimi wrappers override to self.language_model.
        return self.model

    @property
    def sampled_draft_vocab_size(self) -> int | None:
        """The vocabulary the draft's proposal distributions span.

        ``None`` when the compiled graph drafts by argmax. Read off the
        compiled graph's ``next_draft_probs_full`` output rather than the
        config, so the pipeline binds ``draft_probs_full`` exactly when the
        graph declares it.
        """
        outputs = self._spec_decode_model.output_metadata
        # Exports may have one distribution per replica; MAX Serve rejects
        # that layout before constructing its pipeline.
        if len(outputs) < 4:
            return None
        shape = outputs[3].shape
        assert shape is not None and isinstance(shape[-1], int), (
            f"next_draft_probs_full needs a static vocab dim, got {shape}"
        )
        return shape[-1]

    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        model_outputs = self._spec_decode_model.execute(*model_inputs.buffers)
        if len(model_outputs) not in (3, 4):
            raise RuntimeError(
                f"{type(self).__name__} graph returned {len(model_outputs)} "
                "outputs; expected 3 (num_accepted_draft_tokens, next_tokens, "
                "next_draft_tokens) or 4 (+ next_draft_probs_full, under "
                "draft_proposal='sampled')."
            )
        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
            next_draft_probs_full=model_outputs[3]
            if len(model_outputs) == 4
            else None,
        )

    def _next_seed(self) -> Buffer:
        self._seed_counter = getattr(self, "_seed_counter", 0) + 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(self.devices[0])


def _builds_module(arch: SupportedArchitecture) -> bool:
    """Returns whether ``arch`` implements build_module itself.

    Its own pipeline model does, or for a draft with none, its config class.

    This is what tells a pairing that runs through the unified model from
    one that still runs its own spec-decode package. An inherited
    ``build_module`` doesn't count: a subclass may build a different module,
    and a legacy spec-decode model can subclass its target's pipeline model,
    as GLM-5.2's ``UnifiedMTPGlm5_2Model`` does ``Glm5_1Model``.
    """
    builder = _module_builder(arch)
    return (
        arch.pipeline_model is None or issubclass(builder, PipelineModel)
    ) and ("build_module" in vars(builder))


def _takes_images(
    target: SupportedArchitecture, draft: SupportedArchitecture
) -> bool:
    """Returns whether a pairing takes images.

    It does when its target's own pipeline model builds a vision graph and
    its draft takes images too.
    """
    return (
        "build_vision_graph" in vars(target.pipeline_model)
        and InputModality.IMAGE in draft.input_modalities
    )


def _batching(arch: SupportedArchitecture) -> type[Any]:
    """Returns the batch processor ``arch``'s pipeline model batches with."""
    batching = arch.batching or _pipeline_model(arch).batch_processor_cls
    assert batching is not None
    return batching


def _encodes_in_batching(
    target: SupportedArchitecture, draft: SupportedArchitecture
) -> bool:
    """Returns whether a pairing's target encodes images in its batching.

    It does so in its own batch processor rather than through the pipeline's
    vision cache, and that batch processor then prepares each batch of the
    pairing.
    """
    return _takes_images(target, draft) and not issubclass(
        _batching(target), SupportsVisionEncoding
    )


def spec_decode_architecture(
    target: str,
    method: SpeculativeMethod | None,
    draft_arch: str | None,
) -> SupportedArchitecture | None:
    """Returns the fused architecture for a target and draft, if they pair.

    A target pairs with a draft when a proposer runs the method and both
    architectures' pipeline models build their modules. Without a draft
    checkpoint, the draft is the one the target's checkpoint ships for the
    method, if any. ``None`` otherwise, which leaves the pairing to the
    legacy override chain.

    The fused architecture is the target's own, changed only where the
    unified graph differs: it takes text only unless the target builds a
    vision graph and the draft takes images, sizes one KV tree for the target
    and the draft and counts the draft's weights beside the target's, and
    drafts at the width the draft checkpoint was trained for. It runs on
    more than one device only where both architectures do, and is captured
    as a device graph only where the draft's graph allows it. A target that
    checkpoints a recurrent state serves with prefix caching off: the state
    checkpoints only at a page boundary, which a step's accepted drafts move
    past.
    """
    if (
        method is None
        or method not in proposers.SEQUENTIAL_METHODS | proposers.BLOCK_METHODS
    ):
        return None
    # The target first, so a target that can't pair never imports its
    # draft's architecture.
    base = ARCH_LOOKUP.find(target)
    # A draft-only architecture has no pipeline model to serve as the target.
    if base is None or base.pipeline_model is None or not _builds_module(base):
        return None
    draft_arch = draft_arch or base.checkpoint_drafts.get(method)
    if draft_arch is None:
        return None
    draft = ARCH_LOOKUP.find(base.draft_architecture(method, draft_arch))
    if draft is None or not _builds_module(draft):
        return None
    vision = _takes_images(base, draft)
    # A target that encodes its images in its own batching reserves no
    # vision cache.
    vision_cache = vision and not _encodes_in_batching(base, draft)
    return replace(
        base,
        name=f"{base.name}+{method}",
        pipeline_model=(
            _SpecDecodeVisionCachePipelineModel
            if vision_cache
            else _SpecDecodeVisionPipelineModel
            if vision
            else SpecDecodePipelineModel
        ),
        batching=_BatchProcessor,
        cascade_pipeline_factory=None,
        supports_device_graph_capture=draft.supports_device_graph_capture,
        required_arguments=(
            {**base.required_arguments, "enable_prefix_caching": False}
            if base.checkpoints_recurrent_state
            else base.required_arguments
        ),
        config=(
            _SpecDecodeVisionArchConfig
            if vision_cache
            else SpecDecodeArchConfig
        ),
        memory_planner=_MemoryPlanner,
        context_type=base.context_type if vision else TextContext,
        input_modalities=(
            base.input_modalities if vision else {InputModality.TEXT}
        ),
        multi_gpu_supported=(
            base.multi_gpu_supported and draft.multi_gpu_supported
        ),
        checkpoint_draft_width=draft.checkpoint_draft_width,
        # The block driver commits per row, so a prefill row riding a decode
        # batch commits its whole chunk.
        supports_spec_decode_mixed_batches=(
            method in proposers.BLOCK_METHODS
            and base.supports_spec_decode_mixed_batches
        ),
    )


@dataclass(frozen=True)
class _Pairing:
    """The target and draft architectures a resolved config pairs."""

    target_arch: str
    target: SupportedArchitecture
    draft: SupportedArchitecture

    @classmethod
    def from_pipeline_config(cls, pipeline_config: PipelineConfig) -> _Pairing:
        target_arch = pipeline_config.models.main_architecture_name
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None
        method = speculative.speculative_method
        assert method is not None
        target = ARCH_LOOKUP.find(target_arch)
        assert target is not None
        draft_arch = (
            target.draft_architecture(
                method, draft_model.huggingface_config.architectures[0]
            )
            if draft_model is not None
            else target.checkpoint_drafts.get(method)
        )
        assert draft_arch is not None
        draft = ARCH_LOOKUP.find(draft_arch)
        assert draft is not None
        return cls(target_arch, target, draft)


def _takes_distributed_inputs(*parts: object) -> bool:
    """Returns whether the verify or proposer reads the distributed inputs."""
    return any(getattr(p, "takes_distributed_inputs", False) for p in parts)


def _graph_name(pairing: _Pairing, method: str) -> str:
    """Returns the fused graph's name.

    A target whose engine finds its graph by name keeps that name, under the
    draft's architecture where two drafts of one method differ, or under the
    method.
    """
    names = pairing.target.spec_decode_graph_names
    return names.get(pairing.draft.name) or names.get(
        method, f"unified_{method}"
    )


def _draft_checkpoint(pipeline_config: PipelineConfig) -> MAXModelConfig:
    """Returns the draft's own checkpoint, or the target's that ships it."""
    return pipeline_config.draft_model or pipeline_config.model


def _pipeline_model(
    arch: SupportedArchitecture,
) -> type[PipelineModel[Any]]:
    model = arch.pipeline_model
    assert model is not None and issubclass(model, PipelineModel)
    return model


def _module_builder(arch: SupportedArchitecture) -> Any:
    """Returns what builds ``arch``'s module.

    That is its pipeline model, or its config class for a draft that has
    none.
    """
    return arch.pipeline_model or arch.config


def _config_cls(arch: SupportedArchitecture) -> type[Any]:
    """Returns the arch config class ``arch`` builds its module from."""
    if arch.pipeline_model is None:
        return arch.config
    config_cls = _pipeline_model(arch).model_config_cls
    assert config_cls is not None
    return config_cls


def _draft_width(pipeline_config: PipelineConfig) -> int:
    assert pipeline_config.speculative is not None
    return pipeline_config.speculative.draft_width


def _with_num_draft_tokens(
    params: CacheLeafParamInterface, num_draft_tokens: int
) -> CacheLeafParamInterface:
    """Stamps ``num_draft_tokens`` on every attention leaf of a KV params tree.

    A recurrent state has no draft slots.
    """
    if isinstance(params, KVCacheParams):
        return replace(params, num_draft_tokens=num_draft_tokens)
    if not isinstance(params, MultiKVCacheParams):
        return params
    return MultiKVCacheParams.from_params(
        {
            name: _with_num_draft_tokens(child, num_draft_tokens)
            for name, child in params.children.items()
        }
    )


def _first_leaf(params: CacheLeafParamInterface) -> KVCacheParams:
    """The first leaf of a KV params tree, which carries pool-wide fields."""
    while isinstance(params, MultiKVCacheParams):
        params = next(iter(params.children.values()))
    assert isinstance(params, KVCacheParams)
    return params


def _kv_tree(
    pipeline_config: PipelineConfig,
    pairing: _Pairing,
    target_kv: CacheLeafParamInterface,
) -> MultiKVCacheParams:
    """Returns the KV tree the target and the draft share.

    The draft's params are its own config's, on the target's devices, page
    size and cache dtype, which the shared pool requires. A draft whose
    config has none reads and writes the target's leaves, so the tree has
    no draft child.
    """
    pool = _first_leaf(target_kv)
    draft_kv = _config_cls(pairing.draft).construct_kv_params(
        _draft_checkpoint(pipeline_config).huggingface_config,
        pipeline_config,
        list(pool.devices),
        pipeline_config.model.kv_cache.model_copy(
            update={"kv_cache_page_size": pool.page_size}
        ),
        pool.dtype,
    )
    # Every leaf is stamped with the draft width; the pool adds a block
    # draft's extra slot itself. A draft whose block outgrows the width it
    # verifies stamps its own leaf with the wider count it writes.
    width = _draft_width(pipeline_config)
    if draft_kv is None:
        return MultiKVCacheParams.from_params(
            {"target": _with_num_draft_tokens(target_kv, width)}
        )
    width = max(width, _first_leaf(draft_kv).num_draft_tokens)
    return MultiKVCacheParams.from_params(
        {
            "target": _with_num_draft_tokens(target_kv, width),
            "draft": _with_num_draft_tokens(draft_kv, width),
        }
    )


def _check_pairing(
    speculative: SpeculativeConfig, draft_config: Any, target_config: Any
) -> None:
    """Raises if the draft can't run beside the target.

    A draft embeds the target's tokens and drafts in the target's hidden
    space. A block draft also reads the target layers its config names, off
    a target as deep as the one it was trained against when its config says,
    and fills its block with a mask token from the target's vocabulary. A
    draft that embeds and projects only through its target's has no
    vocabulary of its own to check.

    Args:
        speculative: The speculative config.
        draft_config: The config the draft's pipeline model built it from.
        target_config: The target's HuggingFace text config.
    """
    method = speculative.speculative_method
    num_layers = target_config.num_hidden_layers
    layers = proposers.capture_layers(speculative, draft_config)
    if any(not 0 <= i < num_layers for i in layers):
        raise ValueError(
            f"The {method} draft's target_layer_ids must index the target's"
            f" layers, [0, {num_layers}). Got {list(layers)}."
        )
    # The tap ids stay in range whenever the target is deep enough, so a
    # draft trained on another depth would read the same indices off a
    # different residual stream.
    trained_on = getattr(draft_config, "num_target_layers", None)
    if trained_on is not None and trained_on != num_layers:
        raise ValueError(
            f"The {method} draft was trained against a target of"
            f" {trained_on} layers, but this target has {num_layers}."
        )
    if draft_config.hidden_size != target_config.hidden_size:
        raise ValueError(
            f"The {method} draft's hidden_size must match the target's. Got"
            f" draft={draft_config.hidden_size}"
            f" target={target_config.hidden_size}."
        )
    vocab_size = target_config.vocab_size
    draft_vocab_size = getattr(draft_config, "vocab_size", vocab_size)
    if draft_vocab_size != vocab_size:
        raise ValueError(
            f"The {method} draft's vocab_size must match the target's. Got"
            f" draft={draft_vocab_size} target={vocab_size}."
        )
    if speculative.is_block_draft() and not (
        0 <= draft_config.mask_token_id < vocab_size
    ):
        raise ValueError(
            f"The {method} draft's mask_token_id must be in [0, {vocab_size})."
            f" Got {draft_config.mask_token_id}."
        )


def _pairing_stage_error(pipeline_config: PipelineConfig) -> str | None:
    """Returns why the configured pairing can't speculate on a pipeline stage.

    Returns ``None`` if it can. Its target must build stages, and its target
    adapter must verify from a previous stage's hidden states with a draft
    that reads only what the last stage holds.
    """
    pairing = _Pairing.from_pipeline_config(pipeline_config)
    speculative = pipeline_config.speculative
    assert speculative is not None
    method = speculative.speculative_method
    assert method is not None
    if issubclass(
        _pipeline_model(pairing.target), PipelinedModel
    ) and spec_adapters.speculates_on_last_stage(pairing.target_arch, method):
        return None
    return (
        f"A {pairing.target_arch} target does not speculate with {method} on"
        " a pipeline stage"
    )


def _stage_error(pipeline_config: PipelineConfig) -> str | None:
    """Returns why the fused graph can't build its config's pipeline stage.

    Returns ``None`` if it can, or if the config builds the whole model. The
    draft reads what only the last stage holds, and the previous stage
    hands on the merged window's hidden states, so only the last of two or
    more stages speculates.
    """
    stage = pipeline_config.model.pipeline_stage
    if stage is None:
        return None
    if stage.is_first or not stage.is_last:
        return (
            "A fused speculative graph builds only the last of two or more"
            f" pipeline stages, not stage {stage.index} of {stage.num_stages}"
        )
    return _pairing_stage_error(pipeline_config)


def _stage_boundary_input_types(
    stage: PipelineStage | None,
) -> list[TensorType]:
    """Returns a later pipeline stage's boundary inputs.

    They are the hidden states the previous stage returns, which cover the
    merged verify window: earlier stages compute a row's drafts as tokens. The
    previous stage traced them over its own tokens' dim, so they take the dim
    the merge gives the window here.
    """
    if stage is None or stage.boundary_types is None:
        return []
    return [
        TensorType(leaf.dtype, ["merged_seq_len", *leaf.shape[1:]], leaf.device)
        for leaf in tree.leaves(stage.boundary_types)
    ]


@dataclass(kw_only=True)
class SpecDecodeArchConfig:
    """The target config and the KV tree the target and draft share.

    This is the arch config the memory planner and batch processor read, so
    the KV params it reports cover the draft as well as the target.
    """

    target: Any
    target_arch: str
    """The target checkpoint's architecture name."""
    draft_architecture: SupportedArchitecture
    kv_params: MultiKVCacheParams
    target_memory_planner: type[MemoryPlanner] | None = None
    """The target architecture's memory planner."""
    input_spec: SpecDecodeInputTypeSpec | None = None
    """The graph's signature, set once the graph is built."""

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> SpecDecodeArchConfig:
        """Builds the target config and the KV tree it shares with the draft."""
        del model_config
        pairing = _Pairing.from_pipeline_config(pipeline_config)
        target = _config_cls(pairing.target).initialize(
            pipeline_config, max_seq_len=max_seq_len
        )
        return cls(
            target=target,
            target_arch=pairing.target_arch,
            draft_architecture=pairing.draft,
            kv_params=_kv_tree(
                pipeline_config, pairing, target.get_kv_params()
            ),
            target_memory_planner=pairing.target.memory_planner,
        )

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Derives the KV tree from the target and draft configs' own params."""
        pairing = _Pairing.from_pipeline_config(pipeline_config)
        target = _pipeline_model(pairing.target)
        assert issubclass(target, PipelineModelWithKVCache)
        return _kv_tree(
            pipeline_config,
            pairing,
            target.get_kv_params(
                huggingface_config,
                pipeline_config,
                devices,
                kv_cache_config,
                cache_dtype,
            ),
        )

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV tree covering the target and the draft."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """Returns the target's maximum sequence length."""
        return self.target.get_max_seq_len()

    @property
    def devices(self) -> list[DeviceRef]:
        """Returns the target's devices."""
        return list(self.target.devices)

    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: Any, model_config: MAXModelConfig
    ) -> int:
        """Returns the target architecture's maximum sequence length."""
        return _target_config_cls(huggingface_config).calculate_max_seq_len(
            huggingface_config, model_config
        )


class _SpecDecodeVisionArchConfig(SpecDecodeArchConfig):
    """The config of a pairing whose images go through the vision cache.

    It reports the target's vision-cache facts, so the memory planner
    reserves the cache. Any other pairing reports none, whatever its target
    encodes alone.
    """

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Returns the target's vision cache entry size, or ``0`` if none."""
        target = _target_config_cls(huggingface_config)
        if not issubclass(target, ArchConfigWithVisionCache):
            return 0
        return target.estimate_vision_cache_entry_bytes(huggingface_config)

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """Returns the target's vision cache row, or ``None`` if none."""
        target = _target_config_cls(huggingface_config)
        if not issubclass(target, ArchConfigWithVisionCache):
            return None
        return target.get_vision_cache_row_spec(huggingface_config)


def _target_config_cls(huggingface_config: Any) -> type[Any]:
    """Returns the arch config class of the named target."""
    target = ARCH_LOOKUP.find(huggingface_config.architectures[0])
    assert target is not None
    return _config_cls(target)


class _MemoryPlanner(MemoryPlanner):
    """Plans with the target's planner, counting the draft's weights too.

    A draft from a checkpoint of its own loads beside the target, so the
    cache gets what both leave.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        assert isinstance(config, SpecDecodeArchConfig)
        planner = config.target_memory_planner or MemoryPlanner
        self._target = planner(config.target)

    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Returns the target's weights plus the draft checkpoint's."""
        size = self._target.estimate_weights_size(pipeline_config)
        if pipeline_config.draft_model is not None:
            size += pipeline_config.draft_model.weights_size()
        return size

    def estimate_activation_memory(
        self, pipeline_config: Any, huggingface_config: Any
    ) -> int:
        """Returns the target planner's activation estimate."""
        return self._target.estimate_activation_memory(
            pipeline_config, huggingface_config
        )

    def infer_max_batch_size(
        self, pipeline_config: Any, devices: list[Device], weights_size: int
    ) -> int | None:
        """Returns the target planner's default batch size, if it has one."""
        return self._target.infer_max_batch_size(
            pipeline_config, devices, weights_size
        )

    def estimate_signal_buffer_memory(
        self, pipeline_config: Any, arch_config: Any | None = None
    ) -> int:
        """Returns the target planner's signal-buffer estimate."""
        return self._target.estimate_signal_buffer_memory(
            pipeline_config, arch_config
        )


@dataclass
class _Inputs(UnifiedSpecDecodeInputs):
    """Host inputs, packed in the order the graph's signature declares them."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer] = field(default_factory=list)
    host_input_row_offsets: Buffer | None = None
    """The row offsets on the host, under a distributed signature."""
    data_parallel_splits: Buffer | None = None
    """Each replica's first row, under a distributed signature."""
    batch_context_lengths: list[Buffer] = field(default_factory=list)
    """Each device's context length, under a distributed signature."""
    ep_inputs: tuple[Buffer, ...] = ()
    """Expert parallelism's buffers, under a distributed signature."""
    trailing: list[Buffer] = field(default_factory=list)
    """Host inputs the target's call takes past the canonical tail."""
    spec: SpecDecodeInputTypeSpec | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns the buffers in the order the graph's signature declares."""
        spec = self.spec
        assert spec is not None
        kv = tree.leaves(self.kv_cache_inputs) if self.kv_cache_inputs else ()
        # The pipeline sets the images' embeddings and merge positions on
        # every batch of a target that encodes images.
        vision = (
            (*self.vision_embeddings, *self.vision_scatter_indices)
            if spec.enable_vision
            else ()
        )
        prefix: tuple[Buffer, ...]
        if spec.distributed:
            assert self.host_input_row_offsets is not None
            assert self.data_parallel_splits is not None
            prefix = (
                self.tokens,
                *vision,
                self.input_row_offsets,
                self.host_input_row_offsets,
                self.return_n_logits,
                self.data_parallel_splits,
                *self.signal_buffers,
                *kv,
                *self.batch_context_lengths,
                *self.ep_inputs,
            )
        else:
            signals = self.signal_buffers if spec.include_signal_buffers else []
            prefix = (
                self.tokens,
                *vision,
                self.input_row_offsets,
                self.return_n_logits,
                *signals,
                *kv,
            )
        return (
            *prefix,
            *self._spec_decode_tail_buffers(
                include_in_thinking_phase=spec.include_in_thinking_phase,
            ),
            *self.trailing,
        )


class _BatchProcessor(UnifiedSpecDecodeBatchProcessor[_Inputs]):
    """Ragged batching against the unified graph's built signature.

    Under a distributed signature the target batches the inputs ahead of the
    spec-decode tail itself, since its row offsets on the host,
    data-parallel splits, context lengths and expert-parallel buffers are
    its own to compute. A target that encodes its images in its own batching
    prepares the whole batch, the inputs its call takes past the tail
    included.
    """

    def __init__(
        self, config: SpecDecodeArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self.spec_config = config
        speculative = runtime.pipeline_config.speculative
        assert speculative is not None
        assert speculative.speculative_method is not None
        self._method = speculative.speculative_method
        target = ARCH_LOOKUP.find(config.target_arch)
        assert target is not None
        draft = config.draft_architecture
        self._takes_images = _takes_images(target, draft)
        self._encodes_in_batching = _encodes_in_batching(target, draft)

    @cached_property
    def _target_batching(self) -> Any:
        """The target's batching for distributed or image-encoding targets."""
        config = self.spec_config
        assert config.input_spec is not None, "the graph has not been built"
        if not (config.input_spec.distributed or self._encodes_in_batching):
            return None
        target = ARCH_LOOKUP.find(config.target_arch)
        assert target is not None
        return _batching(target)(config.target, self.runtime)

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Hands expert parallelism's buffers to the target's batching.

        Only a target whose batching takes them, as the pipeline model's own
        wiring does.
        """
        bind = getattr(self._target_batching, "bind_ep_comm_initializer", None)
        if bind is not None:
            bind(initializer)

    def bind_vision_encoder(
        self, vision_model: Model, session: InferenceSession
    ) -> None:
        """Hands the compiled vision graph to the target's batching.

        A batching that encodes with the target's whole config, as Kimi
        K2.5's does, gets it too: the pairing keeps only the finalized
        language config, which has no vision half.
        """
        bind_config = getattr(self._target_batching, "bind_model_config", None)
        if bind_config is not None:
            target = ARCH_LOOKUP.find(self.spec_config.target_arch)
            assert target is not None
            bind_config(
                _config_cls(target).initialize(
                    self.runtime.pipeline_config,
                    max_seq_len=self.runtime.max_seq_len,
                )
            )
        self._target_batching.bind_vision_encoder(
            vision_model=vision_model, session=session
        )

    @property
    def vision(self) -> SupportsVisionEncoding[Any]:
        """Returns the target's batching, which encodes a batch's images."""
        assert isinstance(self._target_batching, SupportsVisionEncoding)
        return self._target_batching

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> _Inputs:
        """Prepares the ragged inputs, and any the target's call adds.

        Raises:
            ValueError: If a text-only pairing's batch carries images.
        """
        config = self.spec_config
        arch, target = config.target_arch, config.target
        assert config.input_spec is not None
        # Any image in context, not only one still to encode: a later
        # prefill chunk of an image prompt encodes nothing.
        # TODO(kevinbi): this error ends the model worker instead of failing
        # one request, because `SchedulerResult.failed` does not cover batch
        # preparation.
        if not self._takes_images and any(
            isinstance(ctx, TextAndVisionContext) and ctx.images
            for batch in replica_batches
            for ctx in batch
        ):
            raise ValueError(
                "This speculative pairing serves text only: its fused graph"
                " encodes no images. Drop --speculative-method to serve"
                " images."
            )
        if self._target_batching is None:
            inputs = super().prepare_initial_token_inputs(
                replica_batches, kv_cache_inputs, return_n_logits
            )
        elif self._encodes_in_batching:
            inputs = self._target_inputs(
                replica_batches, kv_cache_inputs, return_n_logits
            )
        else:
            inputs = self._distributed_inputs(
                replica_batches, kv_cache_inputs, return_n_logits
            )
        # Split off after the target's batching, which may read the tail
        # child's leaves into the inputs it prepares.
        tail_key = spec_adapters.tail_kv(arch)
        if tail_key is not None:
            assert inputs.kv_cache_inputs is not None
            inputs.kv_cache_inputs, tail = _split_tail(
                inputs.kv_cache_inputs, tail_key
            )
            inputs.trailing[:0] = spec_adapters.tail_buffers(arch, tail)
        inputs.trailing += spec_adapters.trailing_buffers(
            arch,
            self._method,
            target,
            [ctx for batch in replica_batches for ctx in batch],
            self.runtime.devices,
        )
        if config.input_spec.enable_vision and not self._takes_images:
            embeddings, indices = self._no_images
            inputs.vision_embeddings = embeddings
            inputs.vision_scatter_indices = indices
        return inputs

    @cached_property
    def _no_images(self) -> tuple[list[Buffer], list[Buffer]]:
        """Returns zero-row vision inputs, for an encoder MAX doesn't run."""
        spec = self.spec_config.input_spec
        assert spec is not None and spec.vision_hidden_size is not None
        embeddings = Buffer.zeros(
            shape=[0, spec.vision_hidden_size], dtype=DType.bfloat16
        )
        indices = Buffer.zeros(shape=[0], dtype=DType.int32)
        devices = self.runtime.devices
        return (
            [embeddings.to(device) for device in devices],
            [indices.to(device) for device in devices],
        )

    def _target_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        return_n_logits: int,
    ) -> _Inputs:
        config = self.spec_config
        spec = config.input_spec
        assert spec is not None, "the unified graph has not been built"
        target = self._target_batching.prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        return _Inputs(
            tokens=target.tokens,
            input_row_offsets=target.input_row_offsets,
            return_n_logits=target.return_n_logits,
            signal_buffers=list(target.signal_buffers),
            kv_cache_inputs=target.kv_cache_inputs,
            seed=self._next_seed(self.runtime.devices[0]),
            structured_output=spec.enable_structured_output,
            spec=spec,
            trailing=spec_adapters.batch_trailing_buffers(
                config.target_arch,
                self._method,
                target,
            ),
        )

    def _distributed_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        return_n_logits: int,
    ) -> _Inputs:
        spec = self.spec_config.input_spec
        assert spec is not None, "the unified graph has not been built"
        target = self._target_batching.prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        context_lengths, ep_inputs = spec_adapters.batch_distributed_buffers(
            self.spec_config.target_arch, target
        )
        # Under one replica a target's batch processor may take none of the
        # distributed inputs, as Gemma 4's doesn't, or only some.
        lengths = [
            ctx.tokens.active_length for b in replica_batches for ctx in b
        ]
        host_offsets = getattr(target, "host_input_row_offsets", None)
        if host_offsets is None:
            host_offsets = Buffer.from_numpy(
                np.cumsum([0, *lengths], dtype=np.uint32)
            )
        splits = getattr(target, "data_parallel_splits", None)
        if splits is None:
            splits = Buffer.from_numpy(
                np.array([0, len(lengths)], dtype=np.int64)
            )
        if context_lengths is None:
            context_lengths = [
                Buffer.zeros(shape=[1], dtype=DType.int32)
                for _ in self.runtime.devices
            ]
        return _Inputs(
            tokens=target.tokens,
            input_row_offsets=target.input_row_offsets,
            return_n_logits=target.return_n_logits,
            signal_buffers=list(target.signal_buffers),
            host_input_row_offsets=host_offsets,
            data_parallel_splits=splits,
            batch_context_lengths=list(context_lengths),
            ep_inputs=tuple(ep_inputs),
            kv_cache_inputs=target.kv_cache_inputs,
            seed=self._next_seed(self.runtime.devices[0]),
            structured_output=spec.enable_structured_output,
            spec=spec,
            trailing=spec_adapters.batch_trailing_buffers(
                self.spec_config.target_arch,
                self._method,
                target,
            ),
        )

    def release(self, request_id: RequestID) -> None:
        """Frees what the target's batching holds for a finished request."""
        release = getattr(self._target_batching, "release", None)
        if release is not None:
            release(request_id)

    def release_warmup_state(self, request_ids: list[RequestID]) -> None:
        """Frees what the target's batching claimed for a capture probe."""
        release = getattr(self._target_batching, "release_warmup_state", None)
        if release is not None:
            release(request_ids)

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> _Inputs:
        spec = self.spec_config.input_spec
        assert spec is not None, "the unified graph has not been built"
        return _Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=spec.enable_structured_output,
            spec=spec,
        )


class SpecDecodePipelineModel(
    PipelinedModel[TextContext], PipelineModelWithKVCache[TextContext]
):
    """Serves one fused speculative graph for any paired target and draft.

    A pairing whose target adapter speculates on a pipeline stage also
    builds as the last of several stages.
    """

    model_config_cls: ClassVar[type[Any]] = SpecDecodeArchConfig
    batch_processor_cls: ClassVar[type[Any]] = _BatchProcessor

    model: Model
    ep_comm_initializer: EPCommInitializer | None = None
    """Expert parallelism's runtime, which the target and the draft share."""
    drafter_export: object | None = None
    """What the graph bakes in of a draft that records it, which an export's
    sidecar carries (Mach's ``gen_mef`` reads it)."""
    export_input_types: Sequence[TensorType | BufferType] = ()
    """The fused graph's declared inputs. The compiled graph drops their
    devices and symbolic dimensions, which an export's sidecar keeps."""
    export_contract: Mapping[str, object] | None = None
    """What the target's engine checks the export's inputs against, if it
    keeps a record of its own."""

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
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
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        # Serve the tree the memory planner sized the cache from, not the
        # constructor's second derivation of it.
        self.kv_params = self.arch_config_as(SpecDecodeArchConfig).kv_params
        self.model = self.load_model(session)

    @classmethod
    def stage_speculation_error(
        cls, pipeline_config: PipelineConfig
    ) -> str | None:
        """Returns why the configured pairing can't speculate on a stage.

        Returns ``None`` if it can.
        """
        return _pairing_stage_error(pipeline_config)

    @property
    def sampled_draft_vocab_size(self) -> int | None:
        """The vocabulary the draft's proposal distributions span.

        ``None`` when the compiled graph drafts by argmax. Read off the
        compiled graph's ``next_draft_probs_full`` output rather than the
        config, so the pipeline binds ``draft_probs_full`` exactly when the
        graph declares it.
        """
        outputs = self.model.output_metadata
        # Exports may have one distribution per replica; MAX Serve rejects
        # that layout before constructing its pipeline.
        if len(outputs) < 4:
            return None
        shape = outputs[3].shape
        assert shape is not None and isinstance(shape[-1], int), (
            f"next_draft_probs_full needs a static vocab dim, got {shape}"
        )
        return shape[-1]

    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        """Runs the fused graph and names its outputs."""
        model_outputs = self.model.execute(*model_inputs.buffers)
        if len(model_outputs) not in (3, 4):
            raise RuntimeError(
                f"{type(self).__name__} graph returned {len(model_outputs)} "
                "outputs; expected 3 (num_accepted_draft_tokens, next_tokens, "
                "next_draft_tokens) or 4 (+ next_draft_probs_full, under "
                "draft_proposal='sampled')."
            )
        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
            next_draft_probs_full=model_outputs[3]
            if len(model_outputs) == 4
            else None,
        )

    @property
    def routed_expert_row_bound(self) -> int | None:
        """The target's compiled routed-expert row grid, if it sizes one.

        It an export's sidecar records.
        """
        config = self.arch_config_as(SpecDecodeArchConfig)
        return getattr(config.target, "routed_expert_row_bound", None)

    def release(self, request_id: RequestID) -> None:
        """Frees what the target's batching holds for a finished request."""
        self._batching.release(request_id)

    def release_warmup_state(self, request_ids: list[RequestID]) -> None:
        """Frees what the target's batching claimed for a capture probe."""
        self._batching.release_warmup_state(request_ids)

    @property
    def _batching(self) -> _BatchProcessor:
        assert isinstance(self._batch_processor, _BatchProcessor)
        return self._batch_processor

    @cached_property
    def pairing(self) -> _Pairing:
        """Returns the target and draft architectures this config pairs."""
        return _Pairing.from_pipeline_config(self.pipeline_config)

    @cached_property
    def signal_buffers(self) -> list[Buffer]:
        """Returns the signal buffers the target's collectives need."""
        model = self.pairing.target.pipeline_model
        always = model is not None and issubclass(
            model, AlwaysSignalBuffersMixin
        )
        if len(self.devices) <= 1 and not always:
            return []
        return Signals.allocate(self.devices)

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        """Builds the draft and the target, then compiles the fused graph."""
        if error := _stage_error(self.pipeline_config):
            raise ValueError(error)
        config = self.arch_config_as(SpecDecodeArchConfig)
        speculative = self.pipeline_config.speculative
        assert speculative is not None
        method = speculative.speculative_method
        assert method is not None
        pairing = self.pairing

        ep = _EPRuntime(session)
        context = replace(
            ModelLoadContext.from_pipeline_model(self), ep_runtime=ep.start
        )
        draft_paths = _draft_checkpoint(
            self.pipeline_config
        ).resolved_weight_paths()
        draft, draft_config = _module_builder(pairing.draft).build_module(
            replace(
                context,
                weights=load_weights(draft_paths),
                adapter=pairing.draft.weight_adapters.get(
                    weights_format(draft_paths)
                ),
                # A draft without a leaf of its own runs on the target's.
                kv_params=config.kv_params.children.get(
                    "draft", config.kv_params.children["target"]
                ),
            )
        )
        _check_pairing(
            speculative, draft_config, self.huggingface_config.get_text_config()
        )
        self.drafter_export = getattr(draft_config, "drafter_export", None)
        # A draft without its own embedding or head reads the target's, so
        # those weights load once. Drafts declare it as vLLM's do.
        shared = tuple(
            slot
            for slot in ("embed_tokens", "lm_head")
            if not getattr(draft, f"has_own_{slot}", False)
        )
        draft_weights = {
            name: value
            for name, value in draft.state_dict().items()
            if not name.startswith(tuple(f"{slot}." for slot in shared))
        }
        # The target verifies every row the driver scores, and returns the
        # hidden states the draft reads.
        hidden_states = proposers.target_hidden_states(
            speculative, draft_config
        )
        target, target_config = _pipeline_model(pairing.target).build_module(
            replace(
                context,
                return_logits=ReturnLogits.VARIABLE,
                return_hidden_states=hidden_states,
                target_layer_ids=proposers.capture_layers(
                    speculative, draft_config
                ),
                kv_params=config.kv_params.children["target"],
                encodes_images=_takes_images(pairing.target, pairing.draft),
            )
        )
        # The planner sized the cache from this config before finalizing.
        config.target = target_config
        self.ep_comm_initializer = ep.initializer
        # A target that names its embedding and head otherwise hands them to
        # its draft's proposer instead.
        for slot in shared:
            if hasattr(target, slot):
                setattr(draft, slot, getattr(target, slot))

        with CompilationTimer("model") as timer:
            spec = build_spec_graph(
                self.pipeline_config,
                config.kv_params,
                self.device_refs,
                include_signal_buffers=bool(self.signal_buffers),
                target=target,
                target_config=target_config,
                draft=draft,
                draft_config=draft_config,
                hidden_states=hidden_states,
                draft_weights=draft_weights,
            )
            timer.mark_build_complete()
            model = session.load(
                spec.graph, weights_registry=spec.weights_registry
            )
        config.input_spec = spec.input_spec
        self.export_input_types = spec.input_types
        self.export_contract = spec.export_contract

        self._maybe_release_host_weights(model)
        self._wire_batch_processor(model, config)
        return model


@dataclass(frozen=True)
class SpecGraph:
    """A fused spec-decode graph and what its export records."""

    graph: Graph
    weights_registry: dict[str, Any]
    """The loaded weights, empty for a graph built without them."""
    input_spec: SpecDecodeInputTypeSpec
    input_types: Sequence[TensorType | BufferType]
    """The declared inputs. The compiled graph drops their devices and
    symbolic dimensions."""
    export_contract: Mapping[str, object] | None
    """What the target's engine checks the export's inputs against, if it
    keeps a record of its own."""


def build_spec_graph(
    pipeline_config: PipelineConfig,
    kv_params: MultiKVCacheParams,
    devices: Sequence[DeviceRef],
    *,
    include_signal_buffers: bool,
    target: Any,
    target_config: Any,
    draft: Any,
    draft_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_weights: Mapping[str, Any] | None,
) -> SpecGraph:
    """Fuses the target, the draft and a driver into one graph.

    Args:
        pipeline_config: The pipeline config the two were built from.
        kv_params: The KV tree the target and the draft share.
        devices: The devices the target runs on.
        include_signal_buffers: Whether the graph takes signal buffers.
        target: The target's built module.
        target_config: The config it was built from.
        draft: The draft's built module.
        draft_config: The config it was built from.
        hidden_states: The hidden states the target returns.
        draft_weights: The draft's weights, which load beside the target's.
            ``None`` names every weight without loading any, for a graph
            whose weights stay unbound.
    """
    pairing = _Pairing.from_pipeline_config(pipeline_config)
    target_arch = pairing.target_arch
    speculative = pipeline_config.speculative
    assert speculative is not None
    method = speculative.speculative_method
    assert method is not None
    structured_output = pipeline_config.needs_bitmask_constraints
    if structured_output and speculative.synthetic_acceptance_rate is not None:
        raise ValueError(
            "synthetic_acceptance_rate is incompatible with structured"
            " output: the synthetic acceptance path ignores token"
            " bitmasks. Pass --tool-parser none and leave"
            " --enable-structured-output off to benchmark it."
        )
    spec_adapters.check(target_arch, speculative, structured_output)
    if error := _stage_error(pipeline_config):
        raise ValueError(error)
    boundary_types = _stage_boundary_input_types(
        pipeline_config.model.pipeline_stage
    )
    # A reasoning model's drafts may be accepted under the relaxed
    # in-thinking rule.
    thinking = pipeline_config.runtime.reasoning_parser is not None
    # A target that encodes its images in its own batching passes them
    # past the tail, so only one that encodes through the vision cache
    # takes the pipeline's vision inputs. A graph may also take an encoder's
    # output that MAX doesn't compute, for an engine that runs the encoder
    # itself.
    vision = (
        _takes_images(pairing.target, pairing.draft)
        and not _encodes_in_batching(pairing.target, pairing.draft)
    ) or spec_adapters.takes_vision_outputs(target_arch, method, target_config)
    huggingface_config = pipeline_config.model.huggingface_config
    verify = spec_adapters.verify_for(
        target_arch, target, target_config, hidden_states, draft_config
    )
    proposer: Any
    if speculative.is_block_draft():
        proposer = spec_adapters.block_proposer(
            target_arch, verify, draft, draft_config, speculative
        ) or proposers.block_proposer(
            speculative,
            draft,
            draft_config,
            huggingface_config,
            distributed=_takes_distributed_inputs(verify),
        )
    else:
        proposer = spec_adapters.sequential_proposer(
            target_arch, verify, draft, draft_config, speculative
        ) or proposers.sequential_proposer(
            speculative,
            draft,
            draft_config,
            huggingface_config,
            distributed=_takes_distributed_inputs(verify),
        )
    distributed = _takes_distributed_inputs(verify, proposer)
    input_spec = SpecDecodeInputTypeSpec(
        devices=list(devices),
        distributed=distributed,
        data_parallel_degree=(
            pipeline_config.model.data_parallel_degree if distributed else 1
        ),
        include_signal_buffers=include_signal_buffers,
        include_in_thinking_phase=thinking,
        enable_vision=vision,
        vision_hidden_size=(
            huggingface_config.get_text_config().hidden_size if vision else None
        ),
    )
    # Last of all, so an export without them keeps every slot above.
    penalties = (
        LogitPenalties.input_types(devices[0])
        if speculative.logit_penalties
        else []
    )
    if penalties and speculative.is_block_draft():
        raise ValueError(
            "Logit penalties apply only to the verify of a draft that"
            f" proposes one token a step, and {method} drafts a block."
        )
    driver_kwargs: dict[str, Any] = dict(
        target_model=target,
        draft_model=draft,
        input_spec=input_spec,
        speculative_config=speculative,
        enable_structured_output=structured_output,
        use_greedy_acceptance=speculative.use_greedy_acceptance,
        # Sizes the draft distributions under draft_proposal="sampled".
        vocab_size=huggingface_config.get_text_config().vocab_size,
        ep_input_types=spec_adapters.ep_input_types(target_arch, target),
        # A later pipeline stage's boundary comes after the whole signature
        # the model takes unstaged.
        trailing_input_types=[
            *spec_adapters.trailing_input_types(
                target_arch, method, target_config
            ),
            *penalties,
            *boundary_types,
        ],
    )
    nn_model: SequentialDriver[Any] | BlockDriver[Any, Any]
    if speculative.is_block_draft():
        nn_model = BlockDriver(
            verify,
            proposer,
            relaxed_acceptance=thinking,
            # A block may hold more proposals than a step verifies.
            num_speculative_tokens=speculative.draft_width,
            ctx_at_draft_cache_length=getattr(
                proposer, "ctx_at_draft_cache_length", False
            ),
            **driver_kwargs,
        )
    else:
        nn_model = SequentialDriver(verify, proposer, **driver_kwargs)

    weights_registry: dict[str, Any] = {}
    if draft_weights is None:
        for name, weight in nn_model.raw_state_dict().items():
            weight.name = name
    else:
        # The graph names its weights under ``target.`` and ``draft.``, which
        # Mach binds an exported spec MEF's weights by. The draft loads
        # through the driver, apart from the target's names.
        draft_registry = {f"draft.{k}": v for k, v in draft_weights.items()}
        nn_model.load_state_dict(
            draft_registry,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        target_registry = {
            f"target.{k}": v for k, v in target.state_dict().items()
        }
        for name, weight in target.raw_state_dict().items():
            weight.name = f"target.{name}"
        weights_registry = {**target_registry, **draft_registry}

    kv_params = _without_tail(kv_params, spec_adapters.tail_kv(target_arch))
    primary = spec_adapters.primary_kv(target_arch, method)
    input_types = nn_model.input_types(kv_params)
    export_contract = spec_adapters.export_contract(
        target_arch,
        speculative,
        draft_config,
        target_config,
        kv_params,
        input_types,
        structured_output,
    )
    with Graph(
        _graph_name(pairing, method),
        input_types=input_types,
    ) as graph:
        graph_inputs = nn_model.decode_inputs(graph.inputs, kv_params)
        trailing = list(graph_inputs.trailing)
        boundary = trailing[len(trailing) - len(boundary_types) :]
        trailing = trailing[: len(trailing) - len(boundary_types)]
        extra = spec_adapters.extra(
            target_arch, method, target_config, trailing
        )
        if boundary:
            extra[STAGE_BOUNDARY] = [value.tensor for value in boundary]
        kv_tree = graph_inputs.kv_tree
        assert isinstance(kv_tree, Mapping)
        passthrough = _named_leaves(kv_tree["target"])
        primary_leaf = passthrough.pop(primary)
        # A draft without a leaf of its own reads the target's. A draft
        # with several drives the one the target's primary is named for,
        # or its first, and reads the rest under ``draft_``.
        draft_leaf = primary_leaf
        if "draft" in kv_tree:
            draft_leaves = _named_leaves(kv_tree["draft"])
            draft_leaf = draft_leaves.pop(
                primary if primary in draft_leaves else next(iter(draft_leaves))
            )
            passthrough.update(
                (f"draft_{name}", leaves)
                for name, leaves in draft_leaves.items()
            )
        call_kwargs: dict[str, Any] = dict(
            tokens=graph_inputs.tokens,
            input_row_offsets=graph_inputs.input_row_offsets,
            draft_tokens=graph_inputs.draft_tokens,
            kv_collections=primary_leaf,
            draft_kv_collections=draft_leaf,
            passthrough_kv=passthrough,
            return_n_logits=graph_inputs.return_n_logits,
            signal_buffers=graph_inputs.signal_buffers,
            host_input_row_offsets=graph_inputs.host_input_row_offsets,
            data_parallel_splits=graph_inputs.data_parallel_splits,
            batch_context_lengths=graph_inputs.batch_context_lengths,
            ep_inputs=graph_inputs.ep_inputs or None,
            seed=graph_inputs.seed,
            temperature=graph_inputs.temperature,
            top_k=graph_inputs.top_k,
            max_k=graph_inputs.max_k,
            top_p=graph_inputs.top_p,
            min_top_p=graph_inputs.min_top_p,
            in_thinking_phase=graph_inputs.in_thinking_phase,
            pinned_bitmask=graph_inputs.pinned_bitmask,
            wait_payload=graph_inputs.wait_payload,
            device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
            vision_embeddings=graph_inputs.vision_embeddings,
            vision_scatter_indices=graph_inputs.vision_scatter_indices,
            extra=extra,
        )
        if graph_inputs.draft_probs_full is not None:
            call_kwargs["draft_probs_full"] = graph_inputs.draft_probs_full
        if penalties:
            call_kwargs["penalties"] = LogitPenalties.from_inputs(
                [v.tensor for v in trailing[-len(penalties) :]]
            )
        graph.output(*nn_model(**call_kwargs))

    return SpecGraph(
        graph=graph,
        weights_registry=weights_registry,
        input_spec=nn_model.input_spec,
        input_types=input_types,
        export_contract=export_contract,
    )


class _SpecDecodeVisionPipelineModel(SpecDecodePipelineModel):
    """Serves the fused graph of a target that encodes images.

    The target builds its vision graph, which compiles beside the fused
    graph, and the target's batching runs it on each batch's new images, as
    it does when the target serves alone.
    """

    vision_model: Model

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        """Compiles the fused graph, then the target's vision graph.

        A pipeline stage serves text only, since the first stage merges the
        images into the hidden states it hands on, so it compiles no vision
        graph.
        """
        model = super().load_model(session)
        if self.pipeline_config.model.pipeline_stage is not None:
            return model
        with CompilationTimer("vision model") as timer:
            graph, weights_registry = _pipeline_model(
                self.pairing.target
            ).build_vision_graph(ModelLoadContext.from_pipeline_model(self))
            timer.mark_build_complete()
            self.vision_model = session.load(
                graph, weights_registry=weights_registry
            )
        self._batching.bind_vision_encoder(self.vision_model, session)
        return model


class _SpecDecodeVisionCachePipelineModel(_SpecDecodeVisionPipelineModel):
    """Serves a fused graph whose images go through the vision cache.

    The pipeline's vision cache encodes them with the target's batching.
    """

    model_config_cls: ClassVar[type[Any]] = _SpecDecodeVisionArchConfig

    def pack_vision_inputs(
        self,
        selection: Sequence[tuple[Any, Sequence[ImageMetadata]]],
        devices: list[Device],
    ) -> object | None:
        """Packs the selected images as the target's batching does."""
        return self._batching.vision.pack_vision_inputs(selection, devices)

    def vision_execute(
        self,
        selection: Sequence[tuple[Any, Sequence[ImageMetadata]]],
        devices: list[Device],
        packed: object | None,
    ) -> VisionEncodeResult:
        """Encodes the selected images as the target's batching does."""
        return self._batching.vision.vision_execute(selection, devices, packed)

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Returns the target's embeddings for a batch with no new images."""
        return self._batching.vision.empty_vision_embeddings(devices)


@dataclass
class _EPRuntime:
    """Expert parallelism's runtime, which the target and the draft share.

    The first module to start it sizes its buffers. The draft builds first,
    and one that dispatches its own experts through them dispatches at
    least as wide as a quantized target. Never started in virtual device
    mode, where NVSHMEM can't link.
    """

    session: InferenceSession
    initializer: EPCommInitializer | None = None

    def start(self, ep_config: EPConfig) -> None:
        """Starts the runtime once, and gives ``ep_config`` its node ID."""
        if is_virtual_device_mode():
            return
        if self.initializer is None:
            self.initializer = EPCommInitializer(ep_config)
            self.initializer.ep_init(self.session)
        ep_config.node_id = self.initializer.config.node_id
        if ep_config.node_id == -1:
            raise ValueError(
                "EP node ID is not set. Please check if the EP initialization"
                " is successful."
            )


def _without_tail(
    kv_params: MultiKVCacheParams, tail: str | None
) -> MultiKVCacheParams:
    """Returns the KV tree the graph declares among its KV inputs.

    All of it but the target's child it declares in its tail.
    """
    if tail is None:
        return kv_params
    target = kv_params.children["target"]
    assert isinstance(target, MultiKVCacheParams)
    return MultiKVCacheParams.from_params(
        {
            **kv_params.children,
            "target": MultiKVCacheParams.from_params(
                {k: v for k, v in target.children.items() if k != tail}
            ),
        }
    )


def _split_tail(
    kv_inputs: KVCacheInputs[Buffer, Buffer], tail: str
) -> tuple[KVCacheInputs[Buffer, Buffer], Any]:
    """Splits the target's ``tail`` child off a batch's KV inputs.

    The rest stays an ``OrderedDict``, which flattens in declaration order.
    """
    assert isinstance(kv_inputs, Mapping)
    target = kv_inputs["target"]
    assert isinstance(target, Mapping)
    rest = OrderedDict((k, v) for k, v in target.items() if k != tail)
    return (
        OrderedDict(
            (k, rest if k == "target" else v) for k, v in kv_inputs.items()
        ),
        target[tail],
    )


def _named_leaves(node: Any) -> dict[str, list[Any]]:
    """A KV subtree's per-device leaves, keyed by their name in it."""
    leaf = (KVCacheInputsPerDevice, RecurrentStateInputsPerDevice)
    if not isinstance(node, Mapping):
        return {"": list(tree.leaves(node, leaf=leaf))}
    return {
        name: list(tree.leaves(child, leaf=leaf))
        for name, child in node.items()
    }
