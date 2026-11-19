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
"""Exports a pipeline to a ``.m`` pipeline artifact."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import huggingface_hub
from max._kv_cache_ops import MLASparseKVForm
from max.driver import CPU, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import SymbolicDim
from max.nn.common_layers.context_parallel import (
    PCP_MAX_BATCH_CONTEXT,
    PCP_MIN_ROWS_PER_REQUEST,
    PCP_MIN_STEP_TOKENS,
)
from max.nn.kv_cache import (
    CacheLeafParamInterface,
    KVCacheParams,
    KVLeafRegion,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MSAIndexKVCacheParams,
    MSAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.pipelines.context import SamplingParams, SamplingParamsInput
from max.pipelines.kv_cache import PagedKVCacheManagerInterface
from max.pipelines.lib import SupportedArchitecture, memory_ledger
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.graph_specs import (
    GraphRole,
    GraphSpec,
    compile_and_init_graphs,
    compile_roles_separately,
)
from max.pipelines.lib.interfaces import PipelineModel, PipelineModelWithKVCache
from max.pipelines.lib.memory_plan import MemoryPlan
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
)
from max.pipelines.lib.pipeline_variants.text_generation import (
    TextGenerationPipelineInterface,
)
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.lib.tokenizer import resolve_token_id_bound
from max.pipelines.lib.weight_transform import WeightEntry, WeightNaming
from max.pipelines.modeling.types import InputModality, PipelineTokenizer
from max.pipelines.speculative.spec_width_policy import SpecWidthPolicy
from max.pipelines.weights.hf_utils import HuggingFaceRepo
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import BufferLayout, TensorLayout

from .batch_processor import BatchProcessorDims
from .block_diffusion import BlockDiffusionMetadata
from .engine_config import (
    CaptureCacheBound,
    CapturedHostInput,
    EngineCaptureConfig,
    EngineConfig,
    EngineParallelConfig,
    EnginePrefillContextParallel,
    EngineSamplingConfig,
    EngineSpeculativeConfig,
    SpecGraphFacts,
)
from .format_spec import FormatSpec
from .kv_cache import (
    ContextParallel,
    KVCache,
    KVCacheLayout,
    KVCacheQuantization,
    KVPoolLeaf,
    MHAKVCache,
    MLAKVCache,
    MSAIndexKVCache,
    MSAKVCache,
    RecurrentStateCache,
    SparseKVForm,
    StateRegion,
)
from .model_artifact import ModelArtifact
from .overlap_helpers import (
    SpecFeed,
    artifact_helper_builds,
    attention_cache_count,
    compile_helpers,
)
from .pipeline_modules import PipelineModules
from .sampling_defaults import SamplingDefaults
from .serialization import serialize
from .tokenizer import Tokenizer
from .value_types import (
    ARTIFACT_SUFFIX,
    ArtifactError,
    HuggingFaceRef,
    treedef_to_json,
)

logger = logging.getLogger("max.pipelines")


@runtime_checkable
class SupportsCompiledModules(Protocol):
    """A pipeline that keeps each role's compiled graph.

    The text-generation pipelines work this way.
    """

    @property
    def compiled_modules(self) -> Mapping[str, CompiledCallable[..., Any]]:
        """Each role's compiled graph, keyed by role."""
        ...


@runtime_checkable
class SupportsKVManager(Protocol):
    """A pipeline with a KV cache, as the text-generation pipelines have."""

    @property
    def kv_manager(self) -> PagedKVCacheManagerInterface:
        """The manager of the pipeline's KV cache."""
        ...


@runtime_checkable
class SupportsWeightNaming(Protocol):
    """A pipeline whose graphs' weights may be named by rules.

    The text-generation pipelines expose their model's
    :class:`~max.pipelines.lib.WeightEntry` lists this way.
    """

    @property
    def weight_naming(self) -> Mapping[str, Sequence[WeightEntry]]:
        """The entries that name each graph's weights, by role."""
        ...


def pipeline_modules(
    pipeline: SupportsCompiledModules, architecture: str
) -> PipelineModules:
    """Describes a built pipeline's compiled graphs, keyed by role.

    The graphs are the pipeline's ``compiled_modules``: a text-generation
    pipeline's model's ``graph_specs`` roles and its samplers.

    Args:
        pipeline: The built pipeline.
        architecture: The pipeline's architecture, named in errors.

    Returns:
        The graphs, ready to store as the artifact's ``model`` component.

    Raises:
        ArtifactError: If the pipeline registers no ``decoder`` or
            ``spec_decoder`` graph, which is the case until its pipeline
            model implements ``graph_specs``.
    """
    modules = pipeline.compiled_modules
    if not (GraphRole.DECODER in modules or GraphRole.SPEC_DECODER in modules):
        raise ArtifactError(
            f"{architecture} registers no decoder graph by role; its pipeline"
            " model must implement graph_specs before it can be exported"
        )
    # A GraphRole formats as its member name, so key by its value.
    return PipelineModules(
        {
            role.value if isinstance(role, GraphRole) else role: module
            for role, module in modules.items()
        }
    )


def vision_preprocess_modules(
    model: PipelineModel[Any, Any, Any],
) -> dict[str, CompiledCallable[..., Any]]:
    """Compiles a model's image and video preprocessing graphs, keyed by role.

    Serving never builds the graphs, so export compiles them on their own,
    for the CPU, after the pipeline.

    Args:
        model: The built pipeline model.

    Returns:
        The ``vision_preprocess`` and ``vision_preprocess_video`` graphs the
        model has, which may be none.
    """
    specs: dict[str, GraphSpec] = {
        role: spec
        for role, spec in (
            (GraphRole.VISION_PREPROCESS, model.vision_preprocess_spec()),
            (
                GraphRole.VISION_PREPROCESS_VIDEO,
                model.vision_preprocess_video_spec(),
            ),
        )
        if spec is not None
    }
    if not specs:
        return {}
    compiled = compile_and_init_graphs(InferenceSession(devices=[CPU()]), specs)
    return {GraphRole(role).value: compiled[role] for role in specs}


def model_artifact(
    modules: PipelineModules,
    pipeline_model: PipelineModel[Any, Any, Any] | None,
    batch_processor: BatchProcessorDims | None,
    *,
    engine_config: EngineConfig | None = None,
    kv_cache: KVCacheLayout | None = None,
    format_spec: FormatSpec | None = None,
    tokenizer: Tokenizer | None = None,
    sampling_defaults: SamplingDefaults | None = None,
    weight_loader: WeightNaming | None = None,
) -> ModelArtifact:
    """Assembles the artifact of a built pipeline's graphs.

    A pipeline with a ``vision`` graph also carries its model's vision
    metadata and its image and video preprocessing graphs, which a serving
    engine needs to lay out and preprocess media for that graph.

    Args:
        modules: The pipeline's graphs, keyed by role.
        pipeline_model: The text-generation pipeline's model, or ``None``
            for another pipeline.
        batch_processor: The bounds of the decoder's per-step dims.
        engine_config: The settings the pipeline is built with.
        kv_cache: The layout of the pipeline's KV cache, if it has one.
        format_spec: The FormatSpec document of the checkpoint it serves.
        tokenizer: The tokenizer the checkpoint is served with.
        sampling_defaults: The sampling a request that sets none gets.
        weight_loader: The entries that name each graph's weights.

    Returns:
        The artifact to serialize.

    Raises:
        ArtifactError: If the model declares only one of its vision metadata
            and its image preprocessing graph, since a serving engine needs
            both to serve the ``vision`` graph.
    """
    vision = None
    if pipeline_model is not None and GraphRole.VISION in modules.roles:
        preprocess = vision_preprocess_modules(pipeline_model)
        vision = pipeline_model.vision_metadata()
        has_preprocess = GraphRole.VISION_PREPROCESS.value in preprocess
        if has_preprocess != (vision is not None):
            missing = (
                "vision metadata"
                if has_preprocess
                else "vision_preprocess graph"
            )
            raise ArtifactError(
                f"{type(pipeline_model).__qualname__} exports a vision graph"
                f" with no {missing}; a serving engine needs both to serve it"
            )
        modules = PipelineModules({**modules.roles, **preprocess})
    block_diffusion = None
    if pipeline_model is not None and "block_diffusion" in modules.roles:
        metadata = pipeline_model.block_diffusion_metadata()
        if metadata is None:
            raise ArtifactError(
                f"{type(pipeline_model).__qualname__} exports a"
                " block_diffusion graph but records no block-diffusion"
                " contract; a serving engine cannot run its denoise loop"
            )
        block_diffusion = BlockDiffusionMetadata.from_config(**metadata)

    return ModelArtifact(
        model=modules,
        tokenizer=tokenizer,
        format_spec=format_spec,
        engine_config=engine_config,
        kv_cache=kv_cache,
        batch_processor=batch_processor,
        weight_loader=weight_loader,
        vision=vision,
        sampling_defaults=sampling_defaults,
        block_diffusion=block_diffusion,
    )


def ships_overlap_helpers(devices: Sequence[Device]) -> bool:
    """Returns whether an export onto ``devices`` ships the overlap helpers.

    Overlap scheduling runs only on accelerators, so a host-only artifact
    could never load the helpers.

    Args:
        devices: The devices the exported pipeline runs on.

    Returns:
        ``True`` if any device is an accelerator.
    """
    return not all(device.is_host for device in devices)


def _takes_mrope_positions(graph: CompiledCallable[..., Any]) -> bool:
    """Returns whether ``graph`` takes 3-axis (M-RoPE) positions.

    Its input tree names them with an ``mrope_positions`` node, the group
    the serving engine stages them by.
    """

    def holds(node: object) -> bool:
        if isinstance(node, Mapping):
            if node.get("type") == "mrope_positions":
                return True
            return any(holds(child) for child in node.get("children", ()))
        return False

    return holds(treedef_to_json(graph.graph_signature()[1]))


_SHAPE = CapturedHostInput("shape")

CAPTURED_HOST_INPUTS: Mapping[str, CapturedHostInput] = {
    # The KV cache's host scalars and dispatch records, which MAX's KV
    # manager stages from the aligned batch characteristics
    # (``runtime_inputs(batch_characteristics=...)``).
    **{
        f"paged_kv.{name}": _SHAPE
        for name in (
            "max_prompt_length",
            "max_cache_length",
            "page_stride",
            "scales_page_stride",
            "attention_dispatch_metadata",
            "draft_attention_dispatch_metadata",
            "mla_num_partitions",
            "draft_mla_num_partitions",
            "mixed_step_split",
        )
    },
    # Summed context bounds, every row at the aligned cache length.
    "batch_context.length": _SHAPE,
    "batch_context.total_length": _SHAPE,
    # Uniform once every replica is padded to the captured rows.
    "ragged_batch.data_parallel_splits": _SHAPE,
    "ragged_batch.host_row_offsets": _SHAPE,
    "ragged_batch.host_input_row_offsets": _SHAPE,
    "ragged_batch.return_n_logits": _SHAPE,
    # The graph waits on the handshake, then copies the bitmask from its
    # page-locked buffer.
    "token_bitmask.pinned_bitmask": CapturedHostInput("staged"),
    # The fused sampler's batch bounds, pinned at MAX's overlap pipeline's
    # capture values. max_k -1 is the whole vocabulary, the k a row asking for
    # none falls back to; the GPU kernel reads each row's top_p from the
    # device and never min_top_p, so 1.0 drops no row's top-p.
    "spec_decode.max_k": CapturedHostInput("pinned", -1),
    "spec_decode.min_top_p": CapturedHostInput("pinned", 1.0),
}
"""How a replay gets each host input of a group kind a graph may take,
keyed ``<group kind>.<field>``."""


def captured_host_inputs(
    graphs: Iterable[CompiledCallable[..., Any]],
) -> dict[str, CapturedHostInput]:
    """Returns how a replay gets the host inputs of ``graphs``' groups.

    Args:
        graphs: The graphs a serving engine captures.

    Returns:
        The entries of :data:`CAPTURED_HOST_INPUTS` whose group kind one of
        the graphs' input trees holds.
    """
    kinds: set[str] = set()

    def collect(node: object) -> None:
        if isinstance(node, Mapping):
            if isinstance(kind := node.get("type"), str):
                kinds.add(kind)
            for child in node.get("children", ()):
                collect(child)

    for graph in graphs:
        collect(treedef_to_json(graph.graph_signature()[1]))
    return {
        name: how
        for name, how in CAPTURED_HOST_INPUTS.items()
        if name.partition(".")[0] in kinds
    }


def pipeline_overlap_helpers(
    pipeline_config: PipelineConfig,
    pipeline_model: PipelineModel[Any, Any, Any],
    modules: PipelineModules,
    kv_params: CacheLeafParamInterface | None,
) -> dict[str, CompiledCallable[..., Any]]:
    """Compiles the overlap helpers a serving engine may load, by role.

    A speculative pipeline's in-place feeds are sized from its devices, its
    data-parallel degree, the attention caches in its KV tree, the dtype of
    the drafts its ``spec_decoder`` graph returns, and whether that graph
    takes 3-axis positions.

    Args:
        pipeline_config: The pipeline's configuration.
        pipeline_model: The built pipeline model, whose devices the helpers
            compile for.
        modules: The pipeline's graphs, keyed by role.
        kv_params: The pipeline's KV cache parameters, if it has a cache.

    Returns:
        Each helper's graph, keyed by its
        :class:`~.overlap_helpers.HelperRole`, to store beside the
        pipeline's own roles.
    """
    spec_decoder = modules.roles.get(GraphRole.SPEC_DECODER.value)
    spec_feed = None
    if spec_decoder is not None and kv_params is not None:
        outputs = spec_decoder.out_layouts
        spec_feed = SpecFeed(
            tp=len(pipeline_model.devices),
            kv_groups=attention_cache_count(kv_params),
            dp=pipeline_config.model.data_parallel_degree,
            # The graph returns its next drafts third.
            next_draft_dtype=outputs[2].dtype
            if len(outputs) > 2
            else DType.int64,
            positions=_takes_mrope_positions(spec_decoder),
        )
    return compile_helpers(
        pipeline_model.devices, artifact_helper_builds(spec_feed)
    )


def batch_processor_dims(
    model: PipelineModelWithKVCache[Any], decoder: CompiledCallable[..., Any]
) -> BatchProcessorDims | None:
    """Bounds each symbolic dim of the decoder's per-step inputs.

    Buffers are the inputs an owner allocates once for the model's lifetime,
    such as KV pages or a state pool, and sizes itself. Every other input is
    staged each step, so the serving engine needs the largest size of its
    dims, which :func:`per_step_dims` lists.

    Args:
        model: The built pipeline model.
        decoder: Its compiled decoder.

    Returns:
        The bounds, or ``None`` when the model has not named its graph
        inputs to its batch processor.
    """
    processor = model.batch_processor
    if processor is None or processor.input_tree is None:
        return None
    return BatchProcessorDims(
        processor.max_dims(per_step_dims(decoder.in_layouts), model.kv_params)
    )


def per_step_dims(layouts: Sequence[TensorLayout]) -> list[str]:
    """Returns each symbolic dim of a non-buffer input, in first use."""
    return list(
        {
            dim.name: None
            for layout in layouts
            if not isinstance(layout, BufferLayout)
            for dim in layout.shape
            if isinstance(dim, SymbolicDim)
        }
    )


def _verify_facts(pipeline: object) -> SpecGraphFacts | None:
    """Returns the fused graph's verify facts off the built pipeline model.

    ``None`` for a pipeline whose model is not a fused spec-decode one.
    """
    if not isinstance(pipeline, TextGenerationPipelineInterface):
        return None
    model = pipeline._pipeline_model
    if not isinstance(model, SpecDecodePipelineModel):
        return None
    return SpecGraphFacts(
        verify_takes_prefill_rows=model.verify_takes_prefill_rows,
        verify_commits_long_rows=model.verify_commits_long_rows,
        successor_replay=model.successor_replay,
    )


def pipeline_parallel_config(
    pipeline_config: PipelineConfig,
    num_devices: int,
    context_parallel: ContextParallel | None = None,
    prefill_context_parallel: bool = False,
) -> EngineParallelConfig:
    """Returns how a pipeline splits its work across its devices.

    The decode context degree and interleave are the caches' split, not the
    configuration's flags: an architecture that builds no DCP caches serves
    every position on every rank whatever the flags ask.

    Args:
        pipeline_config: The pipeline's configuration.
        num_devices: The devices the pipeline runs on.
        context_parallel: How the caches split token positions, or ``None``
            when none splits them.
        prefill_context_parallel: Whether prefill steps split their rows
            over the decode context ranks.

    Returns:
        The engine config's ``parallel`` section.

    Raises:
        ArtifactError: If the replicas don't divide the devices, or the
            degrees don't fit together.
    """
    model = pipeline_config.model
    dp = model.data_parallel_degree
    if dp < 1 or num_devices % dp:
        raise ArtifactError(
            f"{dp} data-parallel replicas do not divide {num_devices} devices"
        )
    dcp = 1 if context_parallel is None else context_parallel.degree
    return EngineParallelConfig(
        dp=dp,
        tp=num_devices // dp,
        ep=pipeline_config.runtime.ep_size,
        dcp=dcp,
        pcp=dcp if prefill_context_parallel else 1,
        dcp_interleave=1
        if context_parallel is None
        else context_parallel.interleave,
    )


def pipeline_speculative_config(
    pipeline_config: PipelineConfig,
    max_batch_size: int,
    verify_facts: SpecGraphFacts | None = None,
) -> EngineSpeculativeConfig | None:
    """Returns a pipeline's speculative-decoding facts.

    The verify widths are the ones the pipeline's width policy can produce;
    a decode step's width of 0, which skips the drafter, is one of them; a
    mixed step's 0, which verifies nothing, is recorded only as its
    ``mixed_verify_width``.

    Args:
        pipeline_config: The pipeline's configuration.
        max_batch_size: The planned batch size per replica, which the width
            schedule covers.
        verify_facts: The fused graph's verify facts, read off the built
            spec-decode pipeline model; ``None`` for a pipeline without one.

    Returns:
        The engine config's ``speculative`` section, or ``None`` for a
        pipeline that doesn't decode speculatively.
    """
    speculative = pipeline_config.speculative
    if speculative is None:
        return None
    # The config keeps a speculative section only when a method is set.
    assert speculative.speculative_method is not None
    facts = (
        SpecGraphFacts(False, False, False)
        if verify_facts is None
        else verify_facts
    )
    depth = speculative.draft_width
    runtime = pipeline_config.runtime
    # A graph that verifies beside prefill rows records its mixed width: the
    # serving engine decides at boot whether to serve mixed steps, and a
    # recorded width of 0 would rule them out for it.
    mixed_steps_verify = (
        runtime.enable_spec_decode_mixed_batches
        or facts.verify_takes_prefill_rows
    )
    policy = SpecWidthPolicy.from_config(
        speculative,
        depth,
        max_batch_size,
        mixed_steps_verify=mixed_steps_verify,
    )
    block = speculative.is_block_draft()
    return EngineSpeculativeConfig(
        speculative_method=speculative.speculative_method,
        num_speculative_tokens=depth,
        draft_proposal=speculative.draft_proposal,
        acceptance_mode=(
            "greedy" if speculative.use_greedy_acceptance else "stochastic"
        ),
        reasoning_parser=pipeline_config.runtime.reasoning_parser,
        draft_query_len=depth if block else 1,
        proposer="block" if block else "sequential",
        verify_widths=tuple(
            sorted(
                {
                    *policy.decode_widths(max_batch_size),
                    *([policy.mixed] if policy.mixed else []),
                }
            )
        ),
        mixed_verify_width=(
            policy.max_mixed_width if facts.verify_takes_prefill_rows else None
        ),
        verify_takes_prefill_rows=facts.verify_takes_prefill_rows,
        verify_commits_long_rows=facts.verify_commits_long_rows,
        successor_replay=facts.successor_replay,
    )


def verify_step_prefill_budget(
    max_batch_input_tokens: int,
    max_batch_size: int,
    num_speculative_tokens: int,
) -> int:
    """The prefill tokens a speculative step leaves room for, per replica.

    The serving engine fills its prefill budget with prompt tokens alone;
    each decode row of a verify step then adds its anchor token and its
    ``num_speculative_tokens`` drafts on top. A step therefore holds the
    budget plus ``max_batch_size * (num_speculative_tokens + 1)`` rows, and
    an expert-parallel graph sizes its dispatch at the compiled capacity, so
    the budget is what the capacity leaves after a full batch of those rows.

    Raises:
        ArtifactError: If the capacity does not hold a full batch of them.
    """
    budget = max_batch_input_tokens - max_batch_size * (
        num_speculative_tokens + 1
    )
    if budget <= 0:
        raise ArtifactError(
            f"the compiled input capacity {max_batch_input_tokens} does not"
            f" hold {max_batch_size} verify rows of"
            f" {num_speculative_tokens + 1} tokens"
        )
    return budget


def pipeline_engine_config(
    pipeline_config: PipelineConfig,
    memory_plan: MemoryPlan,
    *,
    verify_facts: SpecGraphFacts | None = None,
    prefix_caching: bool = True,
    prefill_token_budget: int | None = None,
    prefill_capture_tokens: int = 0,
    capture_cache_bound: CaptureCacheBound = "aligned",
    context_parallel: ContextParallel | None = None,
    prefill_context_parallel: EnginePrefillContextParallel | None = None,
    host_inputs: Mapping[str, CapturedHostInput] | None = None,
) -> EngineConfig:
    """Returns the settings a pipeline is built with, for its artifact.

    The batch size, maximum length, batch token cap and devices come from the
    memory plan the pipeline is built with: the configuration holds only what
    was asked for, which planning resolves or lowers.

    Decode capture covers every batch size up to the planned one, for every
    architecture: MAX Serve's ``supports_device_graph_capture`` is its own
    serving policy, and the engine serving the artifact decides which steps
    it can capture.

    Args:
        pipeline_config: The pipeline's configuration.
        memory_plan: The memory plan the pipeline is built with.
        verify_facts: The fused graph's verify facts, read off the built
            spec-decode pipeline model; ``None`` for a pipeline without one.
        prefix_caching: Whether the architecture serves prefix caching;
            ``False`` where its ``required_arguments`` turn it off.
        prefill_token_budget: The scheduler's prefill tokens per replica,
            excluding draft rows; ``None`` derives them from the compiled
            input capacity: all of it, or for a speculative expert-parallel
            pipeline what :func:`verify_step_prefill_budget` leaves. It
            doesn't change that capacity.
        prefill_capture_tokens: The most tokens a captured prefill step
            holds, capped at the prefill tokens a step takes; 0 captures
            no prefill step.
        capture_cache_bound: The host cache bound a replayed step keeps,
            as the architecture declares it.
        context_parallel: How the caches split token positions across a
            decode context parallel group, or ``None`` when no cache splits
            them. It sets ``parallel.dcp`` and ``parallel.dcp_interleave``.
        prefill_context_parallel: When a prefill step takes the PCP path,
            or ``None`` for graphs without one. With it, ``parallel.pcp`` is
            the decode context degree.
        host_inputs: How a replay gets each host input the captured graphs
            read (:func:`captured_host_inputs`).

    Returns:
        The artifact's ``engine_config`` component.

    Raises:
        ArtifactError: If the plan has no maximum length, as for a pipeline
            with no language model, or the settings don't fit together.
    """
    max_length = memory_plan.planned_max_length
    if max_length is None:
        raise ArtifactError("the memory plan sets no maximum length")
    runtime = pipeline_config.runtime
    sampling = pipeline_config.sampling
    devices = memory_plan.require_device_specs()
    max_batch_size = memory_plan.planned_max_batch_size
    capacity = runtime.max_batch_input_tokens
    parallel = pipeline_parallel_config(
        pipeline_config,
        len(devices),
        context_parallel,
        prefill_context_parallel is not None,
    )
    speculative = pipeline_config.speculative
    if (
        prefill_token_budget is None
        and speculative is not None
        and speculative.speculative_method is not None
        and parallel.ep > 1
    ):
        prefill_token_budget = verify_step_prefill_budget(
            capacity, max_batch_size, speculative.draft_width
        )
    prefill_tokens = (
        capacity if prefill_token_budget is None else prefill_token_budget
    )
    return EngineConfig(
        max_batch_size=max_batch_size,
        max_length=max_length,
        max_batch_input_tokens=capacity,
        prefill_token_budget=prefill_token_budget,
        max_batch_total_tokens=memory_plan.planned_max_batch_total_tokens,
        devices=devices,
        parallel=parallel,
        capture=EngineCaptureConfig(
            max_batch_size=max_batch_size,
            prefill_max_tokens=min(prefill_capture_tokens, prefill_tokens),
            host_inputs=dict(host_inputs or {}),
            cache_bound=capture_cache_bound,
        ),
        sampling=EngineSamplingConfig(
            needs_bitmask_constraints=pipeline_config.needs_bitmask_constraints,
            enable_variable_logits=sampling.enable_variable_logits,
            enable_penalties=sampling.enable_penalties,
        ),
        prefix_caching=prefix_caching,
        speculative=pipeline_speculative_config(
            pipeline_config, max_batch_size, verify_facts
        ),
        prefill_context_parallel=prefill_context_parallel,
    )


def _position_split(kv_cache: KVCacheLayout | None) -> ContextParallel | None:
    """Returns how the attention caches split token positions across ranks.

    ``None`` when none splits them. Every attention cache splits them alike,
    which a serving engine checks against ``parallel`` when it loads.
    """
    if kv_cache is None:
        return None
    return next(
        (
            cache.context_parallel
            for cache in kv_cache.caches
            if isinstance(cache, MHAKVCache | MLAKVCache)
            and cache.context_parallel is not None
        ),
        None,
    )


def _splits_positions(kv_cache: KVCacheLayout | None) -> bool:
    """Returns whether any cache splits token positions across ranks."""
    return _position_split(kv_cache) is not None


def _warn_unhonored_context_parallel(
    pipeline_config: PipelineConfig,
    architecture: str,
    kv_cache: KVCacheLayout | None,
) -> None:
    """Warns when the export asks for context parallelism its caches lack.

    Only an architecture that builds DCP caches reads the flags; any other
    exports unsplit caches, which a serving engine serves unsplit.
    """
    model = pipeline_config.model
    if (
        model.decode_context_parallel_degree > 1
        or model.prefill_context_parallel
    ) and not _splits_positions(kv_cache):
        logger.warning(
            "%s does not split its KV cache by token position, so its"
            " artifact ignores the context parallel flags and serves every"
            " position on every rank.",
            architecture,
        )


def _prefill_context_parallel(
    pipeline_config: PipelineConfig, kv_cache: KVCacheLayout | None
) -> EnginePrefillContextParallel | None:
    """Returns when a prefill step takes the PCP path, for its artifact.

    ``None`` unless the export asks for PCP and its caches split positions:
    PCP splits prefill rows over DCP groups, and only an architecture that
    builds DCP caches builds the PCP path.
    """
    if not pipeline_config.model.prefill_context_parallel:
        return None
    if not _splits_positions(kv_cache):
        return None
    return EnginePrefillContextParallel(
        min_step_tokens=PCP_MIN_STEP_TOKENS,
        min_rows_per_request=PCP_MIN_ROWS_PER_REQUEST,
        max_batch_context=PCP_MAX_BATCH_CONTEXT,
    )


_SPARSE_KV_FORMS: dict[MLASparseKVForm, SparseKVForm] = {
    MLASparseKVForm.PLAIN: "plain",
    MLASparseKVForm.SCALED_FP8: "scaled_fp8",
    MLASparseKVForm.NVFP4: "nvfp4",
}


def pipeline_kv_cache(
    params: CacheLeafParamInterface, unread_dispatch: Collection[str] = ()
) -> KVCacheLayout:
    """Returns the KV cache a pipeline's graphs are built for, for its artifact.

    Each cache in the tree is recorded with every parameter its graphs are
    built with and the pool leaves it occupies, in the order the graphs take
    the caches as inputs and named by its path in the tree.

    Args:
        params: The pipeline's KV cache parameters.
        unread_dispatch: The last path keys of the MLA caches whose kernels
            never read their dispatch record.

    Returns:
        The artifact's ``kv_cache`` component.

    Raises:
        ArtifactError: If the tree holds a cache of a type the component
            doesn't record.
    """
    caches = tuple(
        replace(cache, dispatch_unread=True)
        if isinstance(cache, MLAKVCache)
        and cache.name.rpartition(".")[2] in unread_dispatch
        else cache
        for cache in _caches(params, "")
    )
    recorded = {leaf.leaf_id for cache in caches for leaf in cache.leaves}
    assert recorded == set(params.leaves()), (
        "each cache's pool leaves must be the tree's"
    )
    return KVCacheLayout(page_size=params.page_size, caches=caches)


_MHA_RECORDS: dict[type[CacheLeafParamInterface], type[MHAKVCache]] = {
    MHAKVCacheParams: MHAKVCache,
    MSAKVCacheParams: MSAKVCache,
    MSAIndexKVCacheParams: MSAIndexKVCache,
}


def _caches(params: CacheLeafParamInterface, path: str) -> Iterator[KVCache]:
    if isinstance(params, MultiKVCacheParams):
        for key, child in params.children.items():
            yield from _caches(child, f"{path}.{key}" if path else key)
        return
    # A tree passes each child its path as the prefix of its leaf ids.
    leaves = tuple(
        _pool_leaf(region)
        for region in params.leaves(f"{path}." if path else "").values()
    )
    # Exact types: MSA subclasses MHA, and a subclass this function doesn't
    # know may carry parameters the component doesn't record.
    if (cls := _MHA_RECORDS.get(type(params))) is not None:
        assert isinstance(params, MHAKVCacheParams)
        yield cls(
            name=path,
            dtype=params.dtype,
            num_layers=params.num_layers,
            n_kv_heads=params.n_kv_heads,
            head_dim=params.head_dim,
            v_head_dim=params.v_head_dim,
            window_size=params.window_size,
            allow_kv_head_replication=params.allow_kv_head_replication,
            slots_per_page=_slots_per_page(params),
            stride_alignment=params.stride_alignment,
            scratch=params.scratch,
            kvcache_quant_config=_quantization(params),
            speculative_method=params.speculative_method,
            num_draft_tokens=params.num_draft_tokens,
            leaves=leaves,
            context_parallel=_context_parallel(params),
        )
    elif type(params) is MLAKVCacheParams:
        assert isinstance(params, MLAKVCacheParams)
        yield MLAKVCache(
            name=path,
            dtype=params.dtype,
            num_layers=params.num_layers,
            num_q_heads=params.num_q_heads,
            head_dim=params.head_dim,
            v_head_dim=params.v_head_dim,
            window_size=params.window_size,
            slots_per_page=_slots_per_page(params),
            sparse_indices_stride=params.sparse_indices_stride,
            stride_alignment=params.stride_alignment,
            scratch=params.scratch,
            kvcache_quant_config=_quantization(params),
            speculative_method=params.speculative_method,
            num_draft_tokens=params.num_draft_tokens,
            leaves=leaves,
            sparse_kv_form=None
            if params.sparse_indices_stride is None
            else _SPARSE_KV_FORMS[params.resolved_sparse_kv_form],
            context_parallel=_context_parallel(params),
        )
    elif type(params) is RecurrentStateParams:
        assert isinstance(params, RecurrentStateParams)
        yield RecurrentStateCache(
            name=path,
            regions=tuple(
                StateRegion(
                    leaf_id=region.leaf_id,
                    num_layers=region.num_layers,
                    row_shape=tuple(region.row_shape),
                    dtype=region.dtype,
                    scratch=region.scratch,
                )
                for region in params.regions
            ),
            leaves=leaves,
        )
    else:
        raise ArtifactError(
            f"cannot record the {path!r} KV cache: the kv_cache component"
            f" has no kind for a {type(params).__name__}"
        )


def _pool_leaf(region: KVLeafRegion) -> KVPoolLeaf:
    group = region.group_id
    return KVPoolLeaf(
        leaf_id=region.leaf_id,
        group=group.type,
        window_size=group.window_size if group.is_sliding_window() else None,
        bytes_per_page=region.bytes_per_page,
        stride_alignment=region.stride_alignment,
    )


def _slots_per_page(params: KVCacheParams) -> int:
    # Construction resolves an unset slot count to the page size.
    assert params.slots_per_page is not None
    return params.slots_per_page


def _context_parallel(params: KVCacheParams) -> ContextParallel | None:
    split = params.context_parallel
    if split is None:
        return None
    return ContextParallel(degree=split.degree, interleave=split.interleave)


def _quantization(params: KVCacheParams) -> KVCacheQuantization | None:
    config = params.kvcache_quant_config
    if config is None:
        return None
    return KVCacheQuantization(
        scale_dtype=config.scale_dtype,
        quantization_granularity=config.quantization_granularity,
        fp8_tail_dim=config.fp8_tail_dim,
    )


def pipeline_sampling_defaults(
    pipeline_config: PipelineConfig,
) -> SamplingDefaults:
    """Returns the sampling MAX Serve gives a request that sets none.

    It resolves them with MAX Serve's own call: the server's ``--temperature``
    and ``--top-k``, then the checkpoint's ``generation_config.json``, then
    :class:`~max.pipelines.context.SamplingParams`'s defaults.

    Args:
        pipeline_config: The pipeline's configuration.

    Returns:
        The artifact's ``sampling_defaults`` component.
    """
    runtime = pipeline_config.runtime
    params = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(
            temperature=runtime.temperature, top_k=runtime.top_k
        ),
        sampling_params_defaults=pipeline_config.model.sampling_params_defaults,
    )
    return SamplingDefaults(
        temperature=float(params.temperature),
        # MAX spells "no top-k limit" as a non-positive top_k.
        top_k=params.top_k if params.top_k > 0 else None,
        top_p=float(params.top_p),
        repetition_penalty=float(params.repetition_penalty),
    )


def checkpoint_format_spec(
    arch: SupportedArchitecture | None, repo: HuggingFaceRepo
) -> Path | None:
    """Returns the FormatSpec document the architecture names for a checkpoint.

    Args:
        arch: The pipeline's architecture.
        repo: The checkpoint.

    Returns:
        The document's path, or ``None`` when the architecture names none.
    """
    if arch is None or arch.format_spec is None:
        return None
    if callable(arch.format_spec):
        return arch.format_spec(repo)
    return arch.format_spec


def checkpoint_tokenizer(
    arch: SupportedArchitecture | None,
    repo: HuggingFaceRepo,
    tokenizer: PipelineTokenizer[Any, Any, Any],
    pipeline_config: PipelineConfig | None,
) -> Tokenizer | None:
    """Returns the checkpoint's tokenizer files and the ids it is served with.

    The files are the checkpoint's own, at its revision. The chat templates
    are the ones the pipeline's tokenizer renders, including a
    ``--chat-template`` override.

    Args:
        arch: The pipeline's architecture, which names the tokenizer family.
        repo: The checkpoint.
        tokenizer: The pipeline's tokenizer.
        pipeline_config: The pipeline's configuration, which the vocab size
            and EOS ids are read from.

    Returns:
        The component, or ``None`` for a tokenizer with no Hugging Face
        ``delegate``.
    """
    delegate = getattr(tokenizer, "delegate", None)
    if delegate is None:
        return None
    files: dict[str, HuggingFaceRef] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tiktoken.model",
    ):
        if repo.repo_type == "local":
            if Path(repo.local_path, name).is_file():
                # A Hub repo read from a local snapshot keeps its Hub
                # identity, pinned at the snapshot's commit rather than the
                # requested branch. Only a plain local directory is
                # referenced by its path.
                revision = (
                    repo.revision
                    if repo.local_path == repo.repo_id
                    else Path(repo.local_path).name
                )
                files[name] = HuggingFaceRef(repo.repo_id, revision, name)
            continue
        cached = huggingface_hub.try_to_load_from_cache(
            repo.repo_id, name, revision=repo.revision
        )
        if isinstance(cached, str):
            path = cached
        else:
            try:
                path = huggingface_hub.hf_hub_download(
                    repo.repo_id, name, revision=repo.revision
                )
            except huggingface_hub.errors.EntryNotFoundError:
                continue
        # The cache keeps each file at snapshots/<commit>/<name>, which pins
        # the reference to the commit the weights resolved to.
        files[name] = HuggingFaceRef(repo.repo_id, Path(path).parent.name, name)
    template = delegate.chat_template
    templates: dict[str, str | Path] = (
        {"default": template}
        if isinstance(template, str)
        else dict(template or {})
    )
    named = arch.tokenizer_family if arch is not None else None
    if callable(named):
        named = named(repo)
    family, flavor = named or (None, None)
    vocab_size = resolve_token_id_bound(delegate, pipeline_config)
    return Tokenizer(
        family=family,
        flavor=flavor,
        files=files,
        chat_templates=templates,
        vocab_size=vocab_size,
        sampleable_vocab_size=min(
            max(delegate.get_vocab().values()) + 1, vocab_size
        ),
        eos_token_ids=tuple(sorted(tokenizer.eos_token_ids)),
    )


def _check_batch_context_window(
    memory_plan: MemoryPlan, page_size: int
) -> None:
    """Raises unless the batch context window holds a request at max length.

    A serving engine charges each request's context against
    ``max_batch_total_tokens`` in whole KV pages, so a window whose
    page-aligned floor is below ``max_length`` never admits a request that
    long.

    Raises:
        ArtifactError: If the window cannot hold one request at the planned
            maximum length.
    """
    window = memory_plan.planned_max_batch_total_tokens
    max_length = memory_plan.planned_max_length
    if window is None or max_length is None:
        return
    if window // page_size * page_size < max_length:
        needed = (max_length + page_size - 1) // page_size * page_size
        raise ArtifactError(
            f"max_batch_total_tokens {window} cannot hold one request at"
            f" max_length {max_length} in whole {page_size}-token KV pages;"
            f" set it to at least {needed}"
        )


def export_pipeline(
    pipeline_config: PipelineConfig,
    path: str | os.PathLike[str],
    format_spec: Path | None = None,
    *,
    prefill_token_budget: int | None = None,
    prefill_capture_tokens: int = 0,
) -> Path:
    """Builds the pipeline for a config and writes it as a ``.m`` artifact.

    Builds the pipeline the same way serving does, so the artifact holds the
    graphs a served pipeline compiles, the settings it is built with, the
    FormatSpec document and tokenizer of the checkpoint it serves, the
    sampling MAX Serve gives a request that sets none, the entries that
    name each graph's weights if its naming is rules, and the overlap helper
    graphs a serving engine may load, as roles beside the pipeline's.

    Args:
        pipeline_config: The pipeline's configuration.
        path: Where to write the artifact. ``.m`` is appended unless present.
        format_spec: The FormatSpec document to store, in place of the one
            the architecture names for the checkpoint.
        prefill_token_budget: The scheduler's prefill tokens per replica,
            excluding draft rows; ``None`` derives them from the compiled
            input capacity, which it doesn't change.
        prefill_capture_tokens: The most tokens a captured prefill step
            holds, capped at the prefill tokens a step takes; 0 captures no
            prefill step.

    Returns:
        The path of the written artifact.

    Raises:
        ArtifactError: If the pipeline cannot be exported, or the destination
            exists.
    """
    dest = Path(path)
    if dest.suffix != ARTIFACT_SUFFIX:
        dest = dest.with_name(dest.name + ARTIFACT_SUFFIX)
    # Fail before the build, which can take minutes; serialize checks again.
    if dest.exists():
        raise ArtifactError(f"artifact destination {dest} already exists")
    # An architecture whose artifact graphs differ from MAX Serve's reads
    # this when it initializes its config; the caller's config is unchanged.
    pipeline_config = pipeline_config.model_copy(
        update={"artifact_export": True}
    )
    architecture = pipeline_config.models.main_architecture_name
    task = PIPELINE_REGISTRY.retrieve_pipeline_task(architecture)
    arch = PIPELINE_REGISTRY.architecture_for_config(pipeline_config, task)
    # Export records graphs; probing would load and warm the model twice.
    with compile_roles_separately(), memory_ledger.measuring_off():
        retrieved = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config, task=task
        )
        _check_batch_context_window(
            retrieved.memory_plan,
            pipeline_config.model.kv_cache.kv_cache_page_size,
        )
        pipeline = retrieved.factory()
    if not isinstance(pipeline, SupportsCompiledModules):
        raise ArtifactError(
            f"{architecture} builds a {type(pipeline).__name__}, which does"
            " not record its compiled graphs by role"
        )
    model = pipeline_modules(pipeline, architecture)
    checkpoint_arch = PIPELINE_REGISTRY.retrieve_architecture(
        architecture_name=architecture,
        prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
        task=task,
    )
    if (
        checkpoint_arch is not None
        and InputModality.IMAGE in checkpoint_arch.input_modalities
        and GraphRole.VISION not in model.roles
    ):
        logger.warning(
            "%s takes images, but its artifact holds no vision graph, so a"
            " serving engine serves it text only.",
            architecture,
        )
    decoder = model.roles.get("decoder") or model.roles.get("spec_decoder")
    dims = (
        batch_processor_dims(pipeline._pipeline_model, decoder)
        if isinstance(pipeline, TextGenerationPipelineInterface)
        and decoder is not None
        else None
    )
    repo = pipeline_config.model.huggingface_model_repo
    if format_spec is None:
        format_spec = checkpoint_format_spec(arch, repo)
    naming = (
        pipeline.weight_naming
        if isinstance(pipeline, SupportsWeightNaming)
        else {}
    )
    # Only the graphs the artifact holds are named.
    named_graphs = {
        role: entries for role, entries in naming.items() if role in model.roles
    }
    kv_params = (
        pipeline.kv_manager.params
        if isinstance(pipeline, SupportsKVManager)
        else None
    )
    pipeline_model = (
        pipeline._pipeline_model
        if isinstance(pipeline, TextGenerationPipelineInterface)
        else None
    )
    if pipeline_model is not None and ships_overlap_helpers(
        pipeline_model.devices
    ):
        started = time.monotonic()
        helpers = pipeline_overlap_helpers(
            pipeline_config, pipeline_model, model, kv_params
        )
        if clash := sorted(set(helpers) & set(model.roles)):
            raise ArtifactError(
                f"{architecture} registers graphs under the overlap helper"
                f" roles {clash}"
            )
        logger.info(
            "Compiled overlap helpers %s in %.1f s",
            sorted(helpers),
            time.monotonic() - started,
        )
        model = PipelineModules({**model.roles, **helpers})
    kv_cache = (
        pipeline_kv_cache(
            kv_params, arch.unread_dispatch_caches if arch is not None else ()
        )
        if kv_params is not None
        else None
    )
    _warn_unhonored_context_parallel(pipeline_config, architecture, kv_cache)
    artifact = model_artifact(
        model,
        pipeline_model,
        dims,
        engine_config=pipeline_engine_config(
            pipeline_config,
            retrieved.memory_plan,
            verify_facts=_verify_facts(pipeline),
            prefix_caching=arch is None
            or arch.required_arguments.get("enable_prefix_caching")
            is not False,
            prefill_token_budget=prefill_token_budget,
            prefill_capture_tokens=prefill_capture_tokens,
            capture_cache_bound=arch.capture_cache_bound
            if arch is not None
            else "aligned",
            context_parallel=_position_split(kv_cache),
            prefill_context_parallel=_prefill_context_parallel(
                pipeline_config, kv_cache
            ),
            host_inputs=captured_host_inputs(
                graph
                for role, graph in model.roles.items()
                if role
                in (GraphRole.DECODER.value, GraphRole.SPEC_DECODER.value)
            ),
        ),
        kv_cache=kv_cache,
        format_spec=FormatSpec(format_spec)
        if format_spec is not None
        else None,
        tokenizer=checkpoint_tokenizer(
            arch, repo, retrieved.tokenizer, pipeline_config
        ),
        sampling_defaults=pipeline_sampling_defaults(pipeline_config),
        weight_loader=WeightNaming(named_graphs) if named_graphs else None,
    )
    written = serialize(artifact, dest)
    assert artifact.model is not None
    logger.info(
        "Exported graphs %s to %s", sorted(artifact.model.roles), written
    )
    return written
