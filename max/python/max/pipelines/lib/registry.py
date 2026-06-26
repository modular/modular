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

"""Model registry, for tracking various model variants."""

from __future__ import annotations

import functools
import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from max.experimental.nn import Module
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.pipelines.context import (
    PixelContext,
    TextAndVisionContext,
    TextContext,
)
from max.pipelines.kv_cache.memory_planner import MemoryPlanner
from max.pipelines.modeling.types import (
    EmbeddingsContext,
    InputModality,
    Pipeline,
    PipelineTask,
    PipelineTokenizer,
    ReasoningParser,
    TextGenerationRequest,
)
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .pipeline_executor import PipelineExecutor

from max.driver import load_devices
from max.pipelines.diffusion.pipeline import PixelGenerationPipeline
from max.pipelines.lib._hf_config import load_huggingface_config
from max.pipelines.lib.memory_estimation import MemoryEstimator, _MemoryPlan
from max.pipelines.modeling.config_enums import RopeType, SupportedEncoding
from max.pipelines.weights.hf_utils import (
    HuggingFaceRepo,
    is_diffusion_pipeline,
)

from .embeddings_pipeline import EmbeddingsPipeline
from .interfaces import ArchConfig, ArchConfigWithKVCache, PipelineModel
from .pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from .pipeline_variants.text_generation import TextGenerationPipeline
from .reasoning import get_parser_cls
from .tokenizer import TextTokenizer

logger = logging.getLogger("max.pipelines")

PipelineTypes: TypeAlias = Pipeline[Any, Any]

PipelineModelType: TypeAlias = (
    "type[PipelineModel[Any]] "
    "| type[PipelineExecutor[Any, Any, Any]] "
    "| type[Module[Any, Any]]"
)


def get_pipeline_for_task(
    task: PipelineTask, pipeline_config: PipelineConfig
) -> (
    type[TextGenerationPipeline[TextContext]]
    | type[EmbeddingsPipeline]
    | type[PixelGenerationPipeline[Any]]
    | type[OverlapTextGenerationPipeline[TextContext]]
):
    """Returns the pipeline class for the given task and config.

    Args:
        task: The pipeline task (e.g. text generation, embeddings).
        pipeline_config: Pipeline configuration (may select speculative path).

    Returns:
        The pipeline class to use for this task and config.
    """
    if (
        task == PipelineTask.TEXT_GENERATION
        and pipeline_config.speculative is not None
    ):
        spec_method = pipeline_config.speculative.speculative_method
        if (
            pipeline_config.speculative.is_eagle()
            or pipeline_config.speculative.is_mtp()
            or pipeline_config.speculative.is_dflash()
        ):
            return OverlapTextGenerationPipeline[TextContext]
        else:
            raise ValueError(f"Unsupported speculative method: {spec_method}")
    elif pipeline_config.runtime.enable_overlap_scheduler:
        if task == PipelineTask.TEXT_GENERATION:
            return OverlapTextGenerationPipeline[TextContext]
        raise ValueError(
            f"Overlap scheduler requires the TEXT_GENERATION pipeline task, "
            f"got task={task}."
        )
    elif task == PipelineTask.TEXT_GENERATION:
        return TextGenerationPipeline[TextContext]
    elif task == PipelineTask.EMBEDDINGS_GENERATION:
        return EmbeddingsPipeline
    elif task == PipelineTask.PIXEL_GENERATION:
        return PixelGenerationPipeline
    else:
        raise ValueError(f"Unsupported pipeline task: {task}")


@dataclass(frozen=False)
class SupportedArchitecture:
    """Represents a model architecture configuration for MAX pipelines.

    Defines the components and settings required to
    support a specific model architecture within the MAX pipeline system.
    Each `SupportedArchitecture` instance encapsulates the model implementation,
    tokenizer, supported encodings, and other architecture-specific configuration.

    New architectures should be registered into the :obj:`PipelineRegistry`
    using the :obj:`~PipelineRegistry.register()` method.

    Example:
        .. code-block:: python

            my_architecture = SupportedArchitecture(
                name="MyModelForCausalLM",  # Must match your Hugging Face model class name
                example_repo_ids=[
                    "your-org/your-model-name",  # Add example model repository IDs
                ],
                default_encoding="q4_k",
                supported_encodings={
                    "q4_k",
                    "bfloat16",
                    # Add other encodings your model supports
                },
                pipeline_model=MyModel,
                tokenizer=TextTokenizer,
                context_type=TextContext,
                config=MyModelConfig,  # Architecture-specific config class
                default_weights_format=WeightsFormat.safetensors,
                rope_type="none",
                weight_adapters={
                    WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
                    # Add other weight formats if needed
                },
                multi_gpu_supported=True,  # Set based on your implementation capabilities
                required_arguments={"some_arg": True},
                task=PipelineTask.TEXT_GENERATION,
            )
    """

    name: str
    """The name of the model architecture that must match the Hugging Face model class name."""

    example_repo_ids: list[str]
    """A list of Hugging Face repository IDs that use this architecture for testing and validation purposes."""

    default_encoding: SupportedEncoding
    """The default quantization encoding to use when no specific encoding is requested."""

    # TODO: This should be a set[SupportedEncoding] once we remove the sentinel None value.
    supported_encodings: set[SupportedEncoding]
    """A dictionary of supported quantization encodings."""

    pipeline_model: PipelineModelType
    """The model class that defines the graph structure and execution logic.

    Accepts either a :class:`PipelineModel` subclass (for LLM and other
    token-generation architectures) or a :class:`PipelineExecutor` subclass
    (for newer executor-based architectures such as diffusion pipelines).
    """

    task: PipelineTask
    """The pipeline task type that this architecture supports."""

    tokenizer: Callable[..., PipelineTokenizer[Any, Any, Any]]
    """A callable that returns a `PipelineTokenizer` instance for preprocessing model inputs."""

    default_weights_format: WeightsFormat
    """The weights format expected by the `pipeline_model`."""

    context_type: type[TextContext] | type[EmbeddingsContext]
    """The context class type that this architecture uses for managing request state and inputs.

    This should be a class (not an instance) that implements either the `TextContext`
    or `EmbeddingsContext` protocol, defining how the pipeline processes and tracks requests.
    """

    config: type[ArchConfig]
    """The architecture-specific configuration class for the model.

    This class must implement the :obj:`ArchConfig` protocol, providing an
    :obj:`initialize` method that creates a configuration instance from a
    :obj:`PipelineConfig`. For models with KV cache, this should be a class
    implementing :obj:`ArchConfigWithKVCache` to enable KV cache memory estimation.
    """

    rope_type: RopeType = "none"
    """The type of RoPE (Rotary Position Embedding) used by the model."""

    weight_adapters: dict[WeightsFormat, WeightsAdapter] = field(
        default_factory=dict
    )
    """A dictionary of weight format adapters for converting checkpoints from different formats to the default format."""

    multi_gpu_supported: bool = False
    """Whether the architecture supports multi-GPU execution."""

    input_modalities: set[InputModality] = field(
        default_factory=lambda: {InputModality.TEXT}
    )
    """The set of input modalities this architecture accepts.

    Defaults to text-only. Multimodal architectures should declare all
    supported input types explicitly, e.g.
    ``{InputModality.TEXT, InputModality.IMAGE}`` for vision-language models.
    """

    required_arguments: dict[str, bool | int | float] = field(
        default_factory=dict
    )
    """A dictionary specifying required values for PipelineConfig options."""

    context_validators: list[
        Callable[[TextContext | TextAndVisionContext | PixelContext], None]
    ] = field(default_factory=list)
    """A list of callable validators that verify context inputs before model execution.

    These validators are called during context creation to ensure inputs meet
    model-specific requirements. Validators should raise `InputError` for invalid
    inputs, providing early error detection before expensive model operations.

    .. code-block:: python

        def validate_single_image(context: TextContext | TextAndVisionContext) -> None:
            if isinstance(context, TextAndVisionContext):
                if context.pixel_values and len(context.pixel_values) > 1:
                    raise InputError(f"Model supports only 1 image, got {len(context.pixel_values)}")

        my_architecture = SupportedArchitecture(
            # ... other fields ...
            context_validators=[validate_single_image],
        )
    """

    supports_empty_batches: bool = False
    """Whether the architecture can handle empty batches during inference.

    When set to True, the pipeline can process requests with zero-sized batches
    without errors. This is useful for certain execution modes and expert parallelism.
    Most architectures do not require empty batch support and should leave this as False.
    """

    requires_max_batch_context_length: bool = False
    """Whether the architecture requires a max batch context length to be specified.

    If True and max_batch_context_length is not specified, we will default to
    the max sequence length of the model.
    """

    tool_parser: str | Callable[[HuggingFaceRepo], str] | None = None
    """Optional default tool parser for this architecture.

    Either a registered parser name (str), or a callable that takes the
    model's :class:`HuggingFaceRepo` handle (carrying ``repo_id``,
    ``revision``, ``subfolder``, and ``trust_remote_code``) and returns a
    registered parser name. Use the callable form when one architecture
    name covers multiple checkpoint revisions with different tool-call
    grammars (for example, DeepSeek V3 vs V3.1). The callable is invoked
    once during pipeline config resolution and the resulting string is
    stored on ``runtime.tool_parser``.

    The returned name must correspond to a parser registered via
    :func:`max.pipelines.lib.tool_parsing.register`. When set, the
    pipeline config falls back to this value for ``runtime.tool_parser``
    if the user did not explicitly configure one.

    If None, no tool parser is enabled by default and the serving layer
    falls back to its baseline parser.
    """

    batching: type[Any] | None = None
    """Optional batch processor for input/output handling.

    When set, must be a :class:`~max.pipelines.lib.interfaces.batch_processor.BatchProcessor`
    subclass. The processor class is applied to :attr:`pipeline_model` at
    registration time via :attr:`~max.pipelines.lib.interfaces.pipeline_model.PipelineModel.batch_processor_cls`.
    Ragged text models should subclass
    :class:`~max.pipelines.lib.interfaces.batch_processor.RaggedBatchProcessor`.
    """

    reasoning_parser: str | None = None
    """Optional default reasoning parser name for this architecture.

    The name must correspond to a parser registered via
    :func:`max.pipelines.lib.reasoning.register`. When set, the pipeline
    config will fall back to this value for ``runtime.reasoning_parser`` if
    the user did not explicitly configure one. Different model architectures
    emit reasoning content in different formats (e.g., Kimi K2.5 wraps
    reasoning in ``<think>...</think>``), so the appropriate default is
    architecture-specific.

    If None, no reasoning parser is enabled by default and the user must
    opt in by setting ``runtime.reasoning_parser`` explicitly.
    """

    supports_overlap_scheduler: bool = True
    """Whether this architecture supports auto-enabling the overlap scheduler.

    When ``False``, the overlap scheduler is not auto-enabled for this
    architecture even when otherwise eligible. Users can still force-enable
    via ``--enable-overlap-scheduler --force``.
    """

    supports_device_graph_capture: bool = True
    """Whether this architecture supports auto-enabling device graph capture.

    When ``False``, device graph capture is not auto-enabled for this
    architecture even when otherwise eligible. Users can still force-enable
    via ``--device-graph-capture --force``.
    """

    memory_planner: type[MemoryPlanner] | None = None
    """Optional :class:`~max.pipelines.kv_cache.MemoryPlanner` subclass for
    this architecture.

    When set, ``PipelineConfig`` uses the planner to estimate weight size,
    activation memory, signal-buffer memory, and vision cache entry bytes.
    Autoregressive text-generation models should set this to
    :class:`~max.pipelines.kv_cache.PagedMemoryPlanner` (or a subclass with
    architecture-specific overrides).

    ``None`` means the architecture manages its own memory estimation (e.g.
    diffusion pipelines that skip KV cache estimation entirely).
    """

    pipeline_cls: type | None = None
    """Optional pipeline class overriding the task-based default from
    :func:`get_pipeline_for_task`.

    Most architectures leave this ``None`` and are driven by the standard
    task pipelines. Set it when an architecture needs a bespoke generation
    loop that the stock one-token-per-step
    :class:`~max.pipelines.lib.pipeline_variants.text_generation.TextGenerationPipeline`
    cannot express — for example block-diffusion text generation, which runs
    an encoder pass plus an inner denoising loop and emits a whole token
    block per scheduler step. The value must be a
    :class:`~max.pipelines.lib.pipeline_variants.text_generation.TextGenerationPipeline`
    subclass (or compatible) selected after ``pipeline_config.resolve()``.
    """

    @property
    def tokenizer_cls(self) -> type[PipelineTokenizer[Any, Any, Any]]:
        """Returns the tokenizer class for this architecture."""
        if isinstance(self.tokenizer, type):
            return self.tokenizer
        # Otherwise fall back to PipelineTokenizer.
        return TextTokenizer


class _ValidatedNewContext:
    """Picklable wrapper that applies architecture-level validators.

    Unlike a closure or ``functools.wraps``-decorated function, this plain
    class survives pickling when tokenizers are sent to model-worker
    subprocesses.
    """

    def __init__(
        self,
        tokenizer: PipelineTokenizer[Any, Any, Any],
        validators: list[Callable[..., None]],
    ) -> None:
        self._tokenizer = tokenizer
        self._validators = validators

    async def __call__(self, request: Any) -> Any:
        # Call the original (unwrapped) class method, not the instance
        # attribute, so we always reach the real implementation.
        context = await type(self._tokenizer).new_context(
            self._tokenizer, request
        )
        for validator in self._validators:
            validator(context)
        return context


def _apply_context_validators(
    tokenizer: PipelineTokenizer[Any, Any, Any],
    validators: list[Callable[..., None]],
) -> None:
    """Wraps a tokenizer's new_context to apply architecture-level validators.

    This keeps validation logic out of individual tokenizer classes while
    ensuring validators run automatically after context creation.
    """
    wrapper = _ValidatedNewContext(tokenizer, validators)
    tokenizer.new_context = wrapper  # type: ignore[method-assign]


class _ThinkingRegionNewContext:
    """Wraps ``new_context`` to configure the thinking region on each context.

    When a reasoning parser is registered and the context uses constrained
    decoding (grammar or json_schema), the model may start generation
    inside a reasoning span. This wrapper detects that case and suspends
    grammar enforcement until the reasoning-end token fires.

    The reasoning parser and end-token ID are resolved lazily on the first
    call (async), then cached for subsequent requests.
    """

    def __init__(
        self,
        tokenizer: PipelineTokenizer[Any, Any, Any],
        original_new_context: Callable[..., Any],
        reasoning_parser_name: str,
    ) -> None:
        self._tokenizer = tokenizer
        self._original = original_new_context
        self._parser_name = reasoning_parser_name
        self._parser: ReasoningParser | None = None
        self._end_token_id: int | None = None
        self._resolved = False

    async def __call__(self, request: Any) -> Any:
        context = await self._original(request)

        if not self._resolved:
            # Set immediately — no await between check and set, so no
            # interleaving is possible in asyncio's cooperative model.
            self._resolved = True
            parser_cls = get_parser_cls(self._parser_name)
            if parser_cls is not None:
                self._parser = await parser_cls.from_tokenizer(self._tokenizer)
                self._end_token_id = await parser_cls.reasoning_end_token_id(
                    self._tokenizer
                )

        has_constrained_decoding = (
            context.grammar is not None or context.json_schema is not None
        )
        if (
            self._parser is not None
            and self._end_token_id is not None
            and has_constrained_decoding
            and self._parser.will_reason_after_prompt(
                context.tokens.prompt,
            )
        ):
            context.set_thinking_region(None, [self._end_token_id])
            context.grammar_state._in_thinking_region = True
            context.grammar_state.grammar_enforced = False

        return context


def _apply_thinking_region(
    tokenizer: PipelineTokenizer[Any, Any, Any],
    reasoning_parser_name: str | None,
) -> None:
    """Wraps a tokenizer's ``new_context`` to configure thinking regions.

    No-op when *reasoning_parser_name* is ``None``.
    """
    if reasoning_parser_name is None:
        return
    wrapper = _ThinkingRegionNewContext(
        tokenizer, tokenizer.new_context, reasoning_parser_name
    )
    tokenizer.new_context = wrapper  # type: ignore[method-assign]


def _run_memory_planning(
    pipeline_config: Any,
    arch: Any,
    draft_arch: Any = None,
) -> _MemoryPlan:
    """Runs memory estimation and resolves max_length / max_batch_size.

    Called by ``retrieve_factory`` after ``pipeline_config.resolve()`` has
    completed validation. Returns a :class:`_MemoryPlan` whose
    ``max_batch_size`` is passed to the pipeline constructor.

    Also applies ``max_length`` clamping and ``max_batch_total_tokens``
    defaulting, which depend on memory estimation output.
    """
    model_config = pipeline_config.model

    # Diffusion pipelines and non-PipelineModel architectures skip KV-cache
    # memory estimation.
    if is_diffusion_pipeline(
        model_config.huggingface_model_repo
    ) or not issubclass(arch.pipeline_model, PipelineModel):
        return _MemoryPlan(
            max_batch_size=pipeline_config.runtime.max_batch_size or 1,
            footprint=0,
        )

    devices = load_devices(model_config.device_specs)
    arch_config = arch.config.initialize(
        pipeline_config, model_config=model_config
    )

    if arch.memory_planner is not None:
        planner = arch.memory_planner(arch_config)
        weights_size = planner.estimate_weights_size(pipeline_config)
        activation_size = planner.estimate_activation_memory(
            pipeline_config, model_config.huggingface_config
        )
        signal_buffer_size = planner.estimate_signal_buffer_memory(
            pipeline_config, arch_config
        )
    else:
        # ``memory_planner=None`` is the fallback for architectures not yet
        # wired to a MemoryPlanner. If adding a new architecture that uses a
        # KV cache, set ``memory_planner=PagedMemoryPlanner`` on its
        # ``SupportedArchitecture``.
        weights_size = model_config.weights_size()
        activation_size = 0
        signal_buffer_size = pipeline_config.estimate_signal_buffer_memory(
            arch_config
        )

    plan = MemoryEstimator.estimate_memory_footprint(
        pipeline_config,
        model_config,
        arch_config,
        devices,
        weights_size,
        activation_size,
        signal_buffer_size,
        arch=arch,
        max_batch_size=pipeline_config.runtime.max_batch_size,
    )

    # Clamp max_length to what the KV cache can support.
    if clamped_max_seq_len := MemoryEstimator.max_supported_sequence_length(
        weights_size,
        activation_size,
        model_config,
        devices,
        arch_config,
        signal_buffer_size,
    ):
        if model_config.max_length is None:
            model_config.max_length = clamped_max_seq_len
        elif model_config.max_length > clamped_max_seq_len:
            logging.warning(
                "Clamping max_length from %d to %d due to capacity of KV Cache",
                model_config.max_length,
                clamped_max_seq_len,
            )
            model_config.max_length = clamped_max_seq_len

    # For speculative decoding, clamp max_length to the draft model's limit
    # and zero out its cache memory (it shares the target model's KV cache).
    if draft_arch is not None and pipeline_config.draft_model is not None:
        if (
            pipeline_config.draft_model.kv_cache._available_cache_memory
            is not None
        ):
            raise ValueError(
                "Expected draft model's available_cache_memory to be None"
            )
        pipeline_config.draft_model.kv_cache._available_cache_memory = 0
        draft_arch_config = draft_arch.config.initialize(
            pipeline_config, model_config=pipeline_config.draft_model
        )
        draft_max_seq_len = draft_arch_config.get_max_seq_len()
        if (
            model_config.max_length is not None
            and model_config.max_length > draft_max_seq_len
        ):
            logger.info(
                "Clamping max_length from %d to %d (draft model max sequence length)",
                model_config.max_length,
                draft_max_seq_len,
            )
            model_config.max_length = draft_max_seq_len
            pipeline_config.draft_model.max_length = draft_max_seq_len

    # Validate that architectures requiring chunked prefill have it configured.
    # Must run after max_length is resolved.
    if (
        arch.requires_max_batch_context_length
        and pipeline_config.runtime.max_batch_total_tokens is None
    ):
        logger.warning(
            "Architecture '%s' requires max-batch-total-tokens to be specified "
            "but found None. Defaulting to the max sequence length of the model: %s",
            arch.name,
            model_config.max_length,
        )
        pipeline_config.runtime.max_batch_total_tokens = model_config.max_length

    return plan


class PipelineRegistry:
    """Registry for managing supported model architectures and their pipelines.

    This class maintains a collection of :class:`SupportedArchitecture`
    instances, each defining how a particular model architecture should be
    loaded, configured, and executed.

    .. note::

        Do not instantiate this class directly. Always use the global
        :obj:`PIPELINE_REGISTRY` singleton, which is automatically populated
        with all built-in architectures when you import :mod:`max.pipelines`.

    Use :obj:`PIPELINE_REGISTRY` when you want to:

    - **Register a custom architectures**: Call :meth:`register` to add a new
      MAX model architecture to the registry before loading it.
    - **Query supported models**: Call :meth:`retrieve_architecture` to check
      if a Hugging Face model repository is supported before attempting to load it.
    - **Access cached configs**: Methods like :meth:`get_active_huggingface_config` and
      :meth:`get_active_tokenizer` provide cached access to model configurations and tokenizers.
    """

    def __init__(self, architectures: list[SupportedArchitecture]) -> None:
        # Primary lookup by architecture name
        self.architectures = {arch.name: arch for arch in architectures}
        # Secondary lookup for architectures with duplicate names, keyed by (name, task)
        self._architectures_by_task: dict[
            tuple[str, PipelineTask], SupportedArchitecture
        ] = {}
        # Deferred registrations: architecture name -> list of (module, symbol,
        # package) describing *how* to import the SupportedArchitecture. The
        # module is imported lazily the first time the name is looked up (see
        # register_lazy / _materialize_lazy). A name maps to a list because
        # several modules may register the same name under different tasks.
        self._lazy_architectures: dict[
            str, list[tuple[str, str, str | None]]
        ] = {}
        self._cached_huggingface_tokenizers: dict[
            HuggingFaceRepo, PreTrainedTokenizer | PreTrainedTokenizerFast
        ] = {}
        # Tracks already-imported custom architecture specs so that repeated
        # retrieve_factory() calls don't re-run importlib.import_module and
        # spuriously re-register the same architectures.
        self._imported_custom_arch_specs: set[str] = set()

    def register(
        self,
        architecture: SupportedArchitecture,
        *,
        allow_override: bool = False,
    ) -> None:
        """Add new architecture to registry.

        If multiple architectures share the same name but have different tasks,
        they are registered in a secondary lookup table keyed by (name, task).
        """
        if architecture.batching is not None:
            from .interfaces.pipeline_model import PipelineModel

            pipeline_model_cls = architecture.pipeline_model
            if not isinstance(pipeline_model_cls, type) or not issubclass(
                pipeline_model_cls, PipelineModel
            ):
                raise TypeError(
                    f"Architecture '{architecture.name}' sets batching= but "
                    f"pipeline_model {pipeline_model_cls!r} is not a PipelineModel "
                    "subclass."
                )
            pipeline_model_cls.batch_processor_cls = architecture.batching

        task_key = (architecture.name, architecture.task)

        if architecture.name in self.architectures:
            existing_arch = self.architectures[architecture.name]

            # If same task, this is a true conflict
            if existing_arch.task == architecture.task:
                if not allow_override:
                    raise ValueError(
                        f"Refusing to override existing architecture for '{architecture.name}' "
                        f"with task {architecture.task}"
                    )
                logger.warning(
                    f"Overriding existing architecture for '{architecture.name}' with task {architecture.task}"
                )
                self.architectures[architecture.name] = architecture
                self._architectures_by_task[task_key] = architecture
            else:
                # Different tasks - store both, using task-based lookup
                logger.info(
                    f"Registering multiple architectures with name '{architecture.name}': "
                    f"{existing_arch.task} and {architecture.task}"
                )
                # Move existing arch to task-based lookup if not already there
                existing_key = (existing_arch.name, existing_arch.task)
                if existing_key not in self._architectures_by_task:
                    self._architectures_by_task[existing_key] = existing_arch
                # Add new arch to task-based lookup
                self._architectures_by_task[task_key] = architecture
        else:
            # First registration of this name
            self.architectures[architecture.name] = architecture
            self._architectures_by_task[task_key] = architecture

    def register_lazy(
        self,
        name: str,
        module: str,
        symbol: str,
        *,
        package: str | None = None,
    ) -> None:
        """Records *how* to import an architecture without importing it yet.

        This defers the import of an architecture's
        module until the architecture is actually requested. The real
        :class:`SupportedArchitecture` is imported and registered the first
        time ``name`` is looked up; see :meth:`_materialize_lazy`.

        Args:
            name: The architecture name to expose. Must match the ``name`` of
                the :class:`SupportedArchitecture` that ``module``.``symbol``
                resolves to (including any ``_ModuleV3`` suffix).
            module: Dotted module path to import the architecture from. May be
                ``.``-relative, resolved against ``package``.
            symbol: The attribute on ``module`` holding the
                :class:`SupportedArchitecture`.
            package: Anchor package used to resolve a relative ``module`` path.
        """
        self._lazy_architectures.setdefault(name, []).append(
            (module, symbol, package)
        )

    def _materialize_lazy(self, name: str) -> None:
        """Imports and registers any architectures deferred under ``name``.

        No-op when ``name`` has no pending lazy registrations. The entries are
        removed before importing so a failed or repeated lookup does not retry
        the import.
        """
        entries = self._lazy_architectures.pop(name, None)
        if not entries:
            return
        for module, symbol, package in entries:
            imported = importlib.import_module(module, package)
            self.register(getattr(imported, symbol))

    def all_architectures(self) -> list[SupportedArchitecture]:
        """Returns every registered architecture, importing any deferred ones.

        This forces all lazily-registered architectures to be imported, so it
        is only appropriate for callers that genuinely need the full set (for
        example, listing supported models). Normal lookups should go through
        :meth:`retrieve_architecture`, which imports only what it needs.
        """
        for name in list(self._lazy_architectures):
            self._materialize_lazy(name)
        return list(self.architectures.values())

    def retrieve_architecture(
        self,
        architecture_name: str | None,
        prefer_module_v3: bool = False,
        task: PipelineTask | None = None,
    ) -> SupportedArchitecture | None:
        """Retrieve a registered architecture by name.

        Args:
            architecture_name: The architecture class name to look up
                (e.g. ``"LlamaForCausalLM"`` or ``"FluxPipeline"``).
            prefer_module_v3: Whether to use the eager API architecture variant.
                When ``False`` (default), uses the standard graph API architecture name.
                When ``True``, appends the ``_ModuleV3`` suffix to look up the
                eager API architecture.
            task: Optional task to disambiguate when multiple architectures
                share the same name.

        Returns:
            The matching SupportedArchitecture or None if no match found.
        """
        if architecture_name is None:
            return None
        lookup_name = (
            architecture_name + "_ModuleV3"
            if prefer_module_v3
            else architecture_name
        )

        if arch := self._resolve_architecture(lookup_name, task):
            return arch

        # Fallback: if only one variant exists, use it
        fallback_name = (
            architecture_name + "_ModuleV3"
            if not prefer_module_v3
            else architecture_name
        )
        if arch := self._resolve_architecture(fallback_name, task):
            logger.debug(
                "Falling back from '%s' to '%s' (only one variant registered)",
                lookup_name,
                fallback_name,
            )
            return arch

        logger.debug(
            "optimized architecture not available for '%s' in MAX REGISTRY",
            architecture_name,
        )
        return None

    def get_active_huggingface_config(
        self,
        huggingface_repo: HuggingFaceRepo,
    ) -> PretrainedConfig:
        """Retrieves or creates a cached Hugging Face config for the given model.

        Maintains a cache of Hugging Face configurations to avoid
        reloading them unnecessarily which incurs a Hugging Face Hub API call.
        If a config for the given model hasn't been loaded before, it will
        first try ``AutoConfig.from_pretrained()`` (for transformers models),
        then fall back to loading the raw ``config.json`` and creating a
        ``PretrainedConfig`` via ``from_dict()`` (for diffusers components
        and other non-transformers models).

        Note: The cache key is the HuggingFaceRepo itself, whose hash includes
        trust_remote_code and subfolder, so configs with different settings are
        cached separately.
        For multiprocessing, each worker process has its own registry instance
        with an empty cache, so configs are loaded fresh in each worker.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            The Hugging Face configuration object for the model.

        Raises:
            FileNotFoundError: If no ``config.json`` can be found for the
                given repo/subfolder combination.
        """
        return load_huggingface_config(huggingface_repo)

    def get_active_tokenizer(
        self, huggingface_repo: HuggingFaceRepo
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Retrieves or creates a cached Hugging Face AutoTokenizer for the given model.

        Maintains a cache of Hugging Face tokenizers to avoid
        reloading them unnecessarily which incurs a Hugging Face Hub API call.
        If a tokenizer for the given model hasn't been loaded before, it will
        create a new one using AutoTokenizer.from_pretrained() with the model's
        settings.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The Hugging Face tokenizer for the model.
        """
        if huggingface_repo not in self._cached_huggingface_tokenizers:
            self._cached_huggingface_tokenizers[huggingface_repo] = (
                AutoTokenizer.from_pretrained(
                    huggingface_repo.repo_id,
                    trust_remote_code=huggingface_repo.trust_remote_code,
                    revision=huggingface_repo.revision,
                )
            )

        return self._cached_huggingface_tokenizers[huggingface_repo]

    def _resolve_architecture(
        self, name: str, task: PipelineTask | None = None
    ) -> SupportedArchitecture | None:
        """Look up an architecture by name, optionally disambiguating by task.

        When multiple architectures share the same name, the task parameter
        allows selecting the correct one.

        Args:
            name: The architecture name to look up.
            task: Optional task to disambiguate when multiple architectures
                share the same name.

        Returns:
            The matching SupportedArchitecture, or None if not found.
        """
        # Import any architecture deferred under this name before looking it up.
        if name in self._lazy_architectures:
            self._materialize_lazy(name)
        if task is not None:
            task_key = (name, task)
            if task_key in self._architectures_by_task:
                return self._architectures_by_task[task_key]
        return self.architectures.get(name)

    def retrieve_tokenizer(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> PipelineTokenizer[Any, Any, Any]:
        """Retrieves a tokenizer for the given pipeline configuration.

        Args:
            pipeline_config: Configuration for the pipeline
            override_architecture: Optional architecture override string
            task: Optional pipeline task to disambiguate when multiple
                architectures share the same name but serve different tasks.

        Returns:
            PipelineTokenizer: The configured tokenizer

        Raises:
            ValueError: If no architecture is found
        """
        # MAX pipeline
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                architecture_name=pipeline_config.models.main_architecture_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
                task=task,
            )

        if arch is None:
            raise ValueError(
                f"No architecture found for {pipeline_config.models.main_architecture_name}"
            )

        # Calculate Max Length
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required to initialize tokenizer for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        # Use ArchConfigWithKVCache if available for max_seq_len
        if issubclass(arch.config, ArchConfigWithKVCache):
            arch_config = arch.config.initialize(pipeline_config)
            max_length = arch_config.get_max_seq_len()
        else:
            if not issubclass(arch.pipeline_model, PipelineModel):
                raise TypeError(
                    f"Architecture '{arch.name}' must implement "
                    "ArchConfigWithKVCache or use a PipelineModel "
                    "to calculate max_seq_len."
                )
            max_length = arch.pipeline_model.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            )

        tokenizer: PipelineTokenizer[Any, Any, Any]
        if (
            arch.pipeline_model.__name__ in ("MistralModel", "Phi3Model")
            and arch.tokenizer is TextTokenizer
        ):
            text_tokenizer = cast(type[TextTokenizer], arch.tokenizer)
            tokenizer = text_tokenizer(
                pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=pipeline_config.model.retrieve_chat_template(),
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                chat_template=pipeline_config.model.retrieve_chat_template(),
            )

        return tokenizer

    def _import_custom_architectures(
        self, custom_architectures: list[str]
    ) -> None:
        """Imports custom model modules and registers them in the pipeline registry."""
        import importlib
        import os
        import sys

        for module_spec in custom_architectures:
            if module_spec in self._imported_custom_arch_specs:
                continue
            module_parts = module_spec.split(":")
            if len(module_parts) > 2:
                raise ValueError(
                    f"Custom module spec contains too many colons: {module_spec}"
                )
            elif len(module_parts) == 2:
                module_path, module_name = module_parts
            else:
                module_path = os.path.dirname(module_parts[0])
                module_name = os.path.basename(module_parts[0])
            sys.path.append(module_path)
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                raise ValueError(
                    f"Failed to import custom model from: {module_spec}"
                ) from e

            if not module.ARCHITECTURES or not isinstance(
                module.ARCHITECTURES, list
            ):
                raise ValueError(
                    f"Custom model imported, but did not expose an `ARCHITECTURES` list. Module: {module_spec}"
                )

            for arch in module.ARCHITECTURES:
                self.register(arch, allow_override=True)
            self._imported_custom_arch_specs.add(module_spec)

    def retrieve_factory(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer[Any, Any, Any], Callable[[], PipelineTypes]]:
        """Retrieves the tokenizer and a factory that creates the pipeline instance."""
        tokenizer: PipelineTokenizer[Any, Any, Any]
        pipeline_factory: Callable[[], PipelineTypes]

        # Register any user-supplied custom architectures before the arch lookup.
        self._import_custom_architectures(
            pipeline_config.runtime.custom_architectures
        )

        # Apply the unified spec-decode target-architecture override (e.g.
        # "DeepseekV3ForCausalLM" -> "UnifiedMTPDeepseekV3ForCausalLM") *before*
        # resolving ``arch``, so the resolved architecture passed to
        # ``pipeline_config.resolve()`` is consumed by memory estimation, the
        # overlap scheduler, and parser resolution as the overridden arch — not
        # the stale base arch. Resolving after the override (as the inline block
        # in ``resolve()`` did) regressed all unified spec-decode models (#88511).
        pipeline_config._resolve_speculative_target_architecture()

        # MAX pipeline
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                architecture_name=pipeline_config.models.main_architecture_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
                task=task,
            )

        # Architecture should not be None here, as the engine is MAX.
        if arch is None:
            raise ValueError(
                f"No architecture found for {pipeline_config.models.main_architecture_name}"
            )

        # For speculative decoding, pre-resolve the draft architecture before
        # calling resolve() so config.py never needs a registry import.
        draft_arch = None
        if pipeline_config.draft_model is not None:
            draft_arch_name = pipeline_config.draft_model.architecture_name
            if draft_arch_name is None:
                raise ValueError(
                    f"Cannot determine architecture for draft model "
                    f"'{pipeline_config.draft_model.model_path}': "
                    "no 'architectures' field in HuggingFace config."
                )
            draft_arch = self.retrieve_architecture(
                architecture_name=draft_arch_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
            )
            if not draft_arch:
                if not pipeline_config.runtime.prefer_module_v3:
                    v3_draft = self.retrieve_architecture(
                        architecture_name=draft_arch_name,
                        prefer_module_v3=True,
                    )
                    if v3_draft:
                        raise ValueError(
                            f"MAX-optimized architecture found for draft model "
                            f"'{pipeline_config.draft_model.model_path}', but only the "
                            f"new Module-based implementation is available "
                            f"(architecture: '{v3_draft.name}'). "
                            "Please use the '--prefer-module-v3' flag."
                        )
                raise ValueError(
                    "MAX-Optimized architecture not found for `draft_model`"
                )

        # The unified spec-decode target-architecture override is applied above
        # (before ``arch`` is resolved), so ``arch`` already reflects it here
        # and no post-resolve re-resolution is needed.
        pipeline_config.resolve(
            arch,
            draft_arch=draft_arch,
        )

        # Memory planning: derive sizes, run estimation, resolve max_length.
        # Runs after validation (resolve()) and before the factory is built so
        # that all resolved values are available to the pipeline constructor.
        memory_plan = _run_memory_planning(
            pipeline_config, arch, draft_arch=draft_arch
        )

        # Must be called after memory planning so that max_batch_size is known,
        # and after pipeline_config.resolve() so that enable_overlap_scheduler
        # is set correctly (e.g. forced True when --device-graph-capture is
        # explicitly passed).
        pipeline_config._validate_and_resolve_overlap_scheduler(
            arch=arch, max_batch_size=memory_plan.max_batch_size
        )

        pipeline_class = get_pipeline_for_task(task, pipeline_config)

        # An architecture may declare a custom pipeline class that overrides
        # the task-based default (e.g. block-diffusion text generation).
        # ``arch`` is already finalized above, so its choice wins.
        if arch.pipeline_cls is not None:
            pipeline_class = arch.pipeline_cls

        arch_config = arch.config.initialize(pipeline_config)
        max_length = arch_config.get_max_seq_len()

        # For pixel generation (diffusion models), we don't need HuggingFace transformers config
        if task == PipelineTask.PIXEL_GENERATION:
            # Pixel generation pipelines use a different tokenizer with subfolder parameters
            # Check if there's a secondary tokenizer (tokenizer_2) in the manifest
            has_tokenizer_2 = "tokenizer_2" in pipeline_config.models

            # Use the first component's config for model_path and revision.
            first_config = next(iter(pipeline_config.models.values()))

            # Determine tokenizer max_length based on pipeline type.
            # Default to arch_config.get_max_seq_len(); override per-arch as needed.
            if arch.name in {
                "QwenImagePipeline",
                "QwenImageEditPipeline",
                "QwenImageEditPlusPipeline",
            }:
                # QwenImage uses Qwen2 tokenizer with chat template (34 prefix tokens)
                max_length = 1024 + 34
            tokenizer_kwargs = {
                "model_path": first_config.model_path,
                "pipeline_config": pipeline_config,
                "subfolder": "tokenizer",
                "max_length": max_length,
                "revision": first_config.huggingface_model_revision,
                "trust_remote_code": first_config.trust_remote_code,
            }
            if arch.name in ("Flux2Pipeline", "ZImagePipeline"):
                tokenizer_kwargs["max_length"] = 512

            if has_tokenizer_2:
                tokenizer_kwargs["subfolder_2"] = "tokenizer_2"
                secondary_max_length = getattr(
                    arch_config, "secondary_max_seq_len", None
                )
                if secondary_max_length is None:
                    raise ValueError(
                        "secondary_max_seq_len must be set in ArchConfig if tokenizer_2 is present"
                    )
                tokenizer_kwargs["secondary_max_length"] = secondary_max_length

            # Pass per-architecture default for num_inference_steps
            # when the pipeline class declares one.
            default_steps = getattr(
                arch.pipeline_model, "default_num_inference_steps", None
            )
            if default_steps is not None:
                tokenizer_kwargs["default_num_inference_steps"] = default_steps

            tokenizer = arch.tokenizer(**tokenizer_kwargs)

            # Pixel generation pipeline needs pipeline_config, pipeline_model,
            # and cache_config for FBCache/TaylorSeer optimizations.
            pixel_factory_kwargs: dict[str, Any] = {
                "pipeline_config": pipeline_config,
                "pipeline_model": arch.pipeline_model,
                "cache_config": pipeline_config.runtime.denoising_cache,
            }

            pipeline_factory = cast(
                Callable[[], PipelineTypes],
                functools.partial(pipeline_class, **pixel_factory_kwargs),
            )

            # Cast tokenizer for return (pixel generation tokenizer doesn't have eos)
            typed_tokenizer = cast(
                PipelineTokenizer[Any, Any, Any],
                tokenizer,
            )

            return typed_tokenizer, pipeline_factory

        # Load HuggingFace Config for text generation and other tasks
        huggingface_config = pipeline_config.model.huggingface_config

        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required to initialize pipeline for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        # Old Mistral model like Mistral-7B-Instruct-v0.3 uses LlamaTokenizer
        # and suffers from the whitespace decoding bug. So, we enable the fix
        # for only MistralModel in order to avoid any issues with performance
        # for rest of the models. This can be applied more generically once
        # we have more time verifying this for all the models.
        # More information:
        # https://linear.app/modularml/issue/AIPIPE-197/add-support-for-mistral-7b-instruct-v03
        # TODO: remove this pipeline_model.__name__ check
        if (
            arch.pipeline_model.__name__ in ("MistralModel", "Phi3Model")
            and arch.tokenizer is TextTokenizer
        ):
            text_tokenizer = cast(type[TextTokenizer], arch.tokenizer)
            tokenizer = text_tokenizer(
                pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=pipeline_config.model.retrieve_chat_template(),
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                chat_template=pipeline_config.model.retrieve_chat_template(),
            )

        if arch.context_validators:
            _apply_context_validators(tokenizer, arch.context_validators)

        _apply_thinking_region(
            tokenizer, pipeline_config.runtime.reasoning_parser
        )

        # Cast tokenizer to the proper type for text generation pipeline compatibility
        typed_tokenizer = cast(
            PipelineTokenizer[
                Any, npt.NDArray[np.integer[Any]], TextGenerationRequest
            ],
            tokenizer,
        )

        # For speculative decoding, retrieve draft model's architecture
        factory_kwargs: dict[str, Any] = {
            "pipeline_config": pipeline_config,
            "pipeline_model": arch.pipeline_model,
            "eos_token_id": tokenizer.eos,
            "weight_adapters": arch.weight_adapters,
            "tokenizer": typed_tokenizer,
            "memory_plan": memory_plan,
        }

        pipeline_factory = cast(
            Callable[[], PipelineTypes],
            functools.partial(pipeline_class, **factory_kwargs),
        )

        if tokenizer.eos is None:
            raise ValueError(
                "tokenizer.eos value is None, tokenizer configuration is incomplete."
            )

        return tokenizer, pipeline_factory

    def retrieve_context_type(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextContext] | type[EmbeddingsContext]:
        """Retrieve the context class type associated with the architecture for the given pipeline configuration.

        The context type defines how the pipeline manages request state and inputs during
        model execution. Different architectures may use different context implementations
        that adhere to either the TextContext or EmbeddingsContext protocol.

        Args:
            pipeline_config: The configuration for the pipeline.
            override_architecture: Optional architecture name to use instead of looking up
                based on the model repository.
            task: Optional pipeline task to disambiguate when multiple architectures share
                the same name but serve different tasks.

        Returns:
            The context class type associated with the architecture, which implements
            either the TextContext or EmbeddingsContext protocol.

        Raises:
            ValueError: If no supported architecture is found for the given model repository
                or override architecture name.
        """
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                architecture_name=pipeline_config.models.main_architecture_name,
                prefer_module_v3=pipeline_config.runtime.prefer_module_v3,
                task=task,
            )

        if arch:
            return arch.context_type

        raise ValueError(
            f"No architecture found for {pipeline_config.model.model_path}"
        )

    def retrieve_pipeline_task(
        self, architecture_name: str | None
    ) -> PipelineTask:
        """Retrieves the pipeline task for the given architecture name.

        Args:
            architecture_name: The name of the architecture to look up.

        Returns:
            The task associated with the architecture.

        Raises:
            ValueError: If the architecture supports multiple pipeline tasks
                and the user must specify --task explicitly.
            ValueError: If the architecture is not found in the registry.
        """
        if architecture_name is None:
            raise ValueError(
                "Cannot determine pipeline task: architecture name is unknown. "
                "Please specify --task explicitly."
            )
        # Import any architecture deferred under this name so its task(s) are
        # discoverable below.
        if architecture_name in self._lazy_architectures:
            self._materialize_lazy(architecture_name)
        matching_tasks = [
            arch_task
            for (arch_name, arch_task) in self._architectures_by_task
            if arch_name == architecture_name
        ]
        if len(matching_tasks) > 1:
            if PipelineTask.TEXT_GENERATION in matching_tasks:
                other_tasks = [
                    t
                    for t in matching_tasks
                    if t != PipelineTask.TEXT_GENERATION
                ]
                other_task_list = ", ".join(t.value for t in other_tasks)
                logger.warning(
                    f"Architecture '{architecture_name}' supports multiple"
                    f" pipeline tasks. Defaulting to"
                    f" '{PipelineTask.TEXT_GENERATION.value}'. To use a"
                    f" different task, specify --task with one of:"
                    f" {other_task_list}"
                )
                return PipelineTask.TEXT_GENERATION
            task_list = ", ".join(t.value for t in matching_tasks)
            raise ValueError(
                f"Architecture '{architecture_name}' supports multiple "
                f"pipeline tasks: {task_list}. "
                f"Please specify --task explicitly."
            )
        if len(matching_tasks) == 1:
            return matching_tasks[0]
        if arch := self.architectures.get(architecture_name):
            return arch.task
        raise ValueError(
            f"Architecture '{architecture_name}' not found in registry"
        )

    def retrieve(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer[Any, Any, Any], PipelineTypes]:
        """Retrieves the tokenizer and an instantiated pipeline for the config."""
        tokenizer, pipeline_factory = self.retrieve_factory(
            pipeline_config, task, override_architecture
        )
        return tokenizer, pipeline_factory()

    def reset(self) -> None:
        """Clears all registered architectures (mainly for tests)."""
        self.architectures.clear()
        self._architectures_by_task.clear()
        self._lazy_architectures.clear()


PIPELINE_REGISTRY = PipelineRegistry([])
"""Global registry of supported model architectures and their pipelines.

This singleton is automatically populated with all built-in architectures
when you import :mod:`max.pipelines`.

Use ``PIPELINE_REGISTRY`` to:

- **Register custom architectures**: Call :meth:`~PipelineRegistry.register()`
  to add a new model architecture.
- **Query supported models**: Call
  :meth:`~PipelineRegistry.retrieve_architecture()` to check whether a
  Hugging Face model repository is supported.
- **Access cached configs**: Use
  :meth:`~PipelineRegistry.get_active_huggingface_config()` and
  :meth:`~PipelineRegistry.get_active_tokenizer()` for cached access to model
  configurations and tokenizers.

See :class:`PipelineRegistry` for the full API.
"""
