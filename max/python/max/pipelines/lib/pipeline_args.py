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

"""Flat, user-facing input arguments for a MAX pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from max.config import ConfigFileModel
from max.driver import DeviceSpec
from max.dtype import DType
from max.engine import GPUProfilingMode
from max.pipelines.diffusion.cache import DenoisingCacheConfig
from max.pipelines.kv_cache.config import KVCacheConfig
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.device_specs import (
    _default_device_specs,
    coerce_device_specs_input,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import (
    DEFAULT_MAX_BATCH_INPUT_TOKENS,
)
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.config_enums import (
    PipelineRole,
    RopeType,
    SupportedEncoding,
)
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.speculative.config import SpeculativeConfig
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from max.pipelines.lib.config.config import PipelineConfig


class PipelineArgs(ConfigFileModel):
    """Flat, user-settable input arguments for a pipeline.

    ``PipelineArgs`` is the user-facing input to the pipeline system. It
    holds only scalar fields and a small number of cohesive sub-config
    objects (``kv_cache``, ``lora``, ``speculative``, ``draft_model``).

    Multi-component pipelines (e.g. diffusion) that require a pre-built
    :class:`~max.pipelines.lib.model_manifest.ModelManifest` may pass
    ``models=<manifest>`` to the constructor. That manifest is stored as a
    private override and used verbatim by :meth:`PipelineConfig.from_args`
    instead of constructing one from the flat scalar fields.

    Call :meth:`PipelineConfig.from_args` to obtain a fully-constructed
    :class:`PipelineConfig` ready for architecture-driven resolution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------ #
    # Top-level pipeline fields
    # ------------------------------------------------------------------ #

    model_override: list[str] = Field(
        default_factory=list,
        description=(
            "Per-component overrides for the ModelManifest, in the format "
            "``component.field=value``. Applied before resolution. Repeatable."
        ),
    )

    task: PipelineTask = Field(
        default=PipelineTask.UNDEFINED,
        description=(
            "The pipeline task to run (e.g. ``text_generation``, "
            "``embeddings_generation``). Used to disambiguate architectures "
            "registered under the same name for multiple tasks."
        ),
    )

    debug_verify_replay: bool = Field(
        default=False,
        description=(
            "When ``device_graph_capture`` is enabled, execute eager launch-trace "
            "verification before replay. Intended for debugging only."
        ),
    )

    # ------------------------------------------------------------------ #
    # Fields from MAXModelConfig
    # ------------------------------------------------------------------ #

    model_path: str = Field(
        default="",
        description=(
            "Accepts either a Hugging Face repository ID "
            "or a local path to the model."
        ),
    )

    served_model_name: str | None = Field(
        default=None,
        description=(
            "Optional override for client-facing model name. Defaults to "
            "``model_path``."
        ),
    )

    weight_path: list[Path] = Field(
        default_factory=list,
        description=(
            "Optional path or URL of the model weights to use. "
            "Overrides default weight discovery."
        ),
    )

    quantization_encoding: SupportedEncoding | None = Field(
        default=None,
        description=(
            "Weight encoding type. For GGUF models, the encoding is "
            "auto-detected from the repository when unset."
        ),
    )

    huggingface_model_revision: str = Field(
        default="main",
        description=(
            "Branch or Git revision of Hugging Face model repository to use."
        ),
    )

    huggingface_weight_revision: str = Field(
        default="main",
        description=(
            "Branch or Git revision of Hugging Face weight repository to use."
        ),
    )

    trust_remote_code: bool = Field(
        default=False,
        description=(
            "Whether or not to allow for custom modeling files on Hugging Face."
        ),
    )

    subfolder: str | None = Field(
        default=None,
        description=(
            "Subdirectory within the HuggingFace repo to load config and "
            "weights from."
        ),
    )

    device_specs: list[DeviceSpec] = Field(
        default_factory=_default_device_specs,
        description=("Devices to run inference upon."),
    )

    @field_validator("device_specs", mode="before")
    @classmethod
    def _coerce_device_specs(cls, value: Any) -> list[DeviceSpec]:
        return coerce_device_specs_input(value)

    force_download: bool = Field(
        default=False,
        description=(
            "Whether to force download a given file if it's already present in "
            "the local cache."
        ),
    )

    vision_config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=("Model-specific vision configuration overrides."),
    )

    rope_type: RopeType | None = Field(
        default=None,
        description=(
            "Force using a specific rope type. Only matters for GGUF weights."
        ),
    )

    sliding_window: int | None = Field(
        default=None,
        description=(
            "If set, overrides the model's attention to use a "
            "sliding-window causal mask of this many tokens."
        ),
    )

    enable_echo: bool = Field(
        default=False,
        description="Whether the model should be built with echo capabilities.",
    )

    chat_template: Path | None = Field(
        default=None,
        description=(
            "Optional custom chat template to override the one shipped with the "
            "Hugging Face model config."
        ),
    )

    use_subgraphs: bool = Field(
        default=True,
        description=("Whether to use subgraphs for the model."),
    )

    data_parallel_degree: int = Field(
        default=1,
        description=("Data-parallelism parameter."),
    )

    pool_embeddings: bool = Field(
        default=True,
        description="Whether to pool embedding outputs.",
    )

    max_length: int | None = Field(
        default=None,
        description=("Maximum sequence length the model can process."),
    )

    kv_cache: KVCacheConfig = Field(
        default_factory=KVCacheConfig,
        description="The ``KVCacheConfig`` instance.",
    )

    # ------------------------------------------------------------------ #
    # Fields from PipelineRuntimeConfig
    # ------------------------------------------------------------------ #

    pipeline_role: PipelineRole = Field(
        default="prefill_and_decode",
        description=(
            "Whether the pipeline should serve both a prefill or decode role or both."
        ),
    )

    max_batch_size: int | None = Field(
        default=None,
        description=("Maximum batch size to execute with the model."),
    )

    max_queue_size_tg: int | None = Field(
        default=None,
        description=("Maximum number of requests in decode queue."),
    )

    min_batch_size_tg: int | None = Field(
        default=None,
        description=("Soft floor on the decode batch size."),
    )

    ep_size: int = Field(
        default=1,
        description=("The expert parallelism size."),
    )

    ep_use_allreduce: bool = Field(
        default=False,
        description=(
            "Whether to use allreduce for the cross-device communication in "
            "expert parallelism."
        ),
    )

    eplb_profile: bool = Field(
        default_factory=lambda: os.getenv("MAX_SERVE_EPLB_PROFILE", "").lower()
        in ("1", "true", "yes"),
        description=(
            "When True, enables expert-parallel load balancing (EPLB) MoE "
            "routing histogram profiling in the pipeline."
        ),
    )

    ce_delay_ms: float = Field(
        default=0.0,
        description=(
            "Duration of scheduler sleep prior to starting a prefill batch."
        ),
    )

    enable_prioritize_first_decode: bool = Field(
        default=False,
        description=(
            "When enabled, the scheduler always runs a TG batch immediately "
            "after a CE batch with the same requests."
        ),
    )

    enable_chunked_prefill: bool = Field(
        default=True,
        description=(
            "Enable chunked prefill to split context encoding requests into "
            "multiple chunks based on ``max_batch_input_tokens``."
        ),
    )

    enable_in_flight_batching: bool = Field(
        default=False,
        description=(
            "When enabled, prioritizes token generation by batching it with "
            "context encoding requests."
        ),
    )

    eplb_replicas_per_gpu: int = Field(
        default=0,
        description=(
            "Number of redundant expert replicas to add per GPU when EPLB is active."
        ),
    )

    max_num_steps: int = Field(
        default=1,
        description=(
            "Deprecated. Multi-step pipeline execution is no longer supported."
        ),
    )

    max_batch_input_tokens: int = Field(
        default=DEFAULT_MAX_BATCH_INPUT_TOKENS,
        description=(
            "The target number of un-encoded tokens to include in each batch."
        ),
    )

    use_experimental_kernels: str = Field(
        default=os.environ.get("USE_EXPERIMENTAL_KERNELS", "false"),
        description=(
            "Enables using experimental Mojo kernels with ``max serve``."
        ),
    )

    use_vendor_blas: str = Field(
        default=os.environ.get("MAX_SERVE_USE_VENDOR_BLAS", "false"),
        description=("Enables using vendor BLAS libraries with ``max serve``."),
    )

    use_vendor_ccl: str = Field(
        default=os.environ.get("MAX_SERVE_USE_VENDOR_CCL", "false"),
        description=(
            "Enables using vendor CCL libraries for collective operations."
        ),
    )

    custom_architectures: list[str] = Field(
        default_factory=list,
        description=("Custom architecture implementations to register."),
    )

    execute_empty_batches: bool = Field(
        default=False,
        description=(
            "When enabled, the scheduler runs the model's forward pass even "
            "for an empty batch."
        ),
    )

    max_batch_total_tokens: int | None = Field(
        default=None,
        description=(
            "Ensures the sum of page-aligned context lengths in a batch does "
            "not exceed ``max_batch_total_tokens``."
        ),
    )

    device_graph_capture: bool | None = Field(
        default=None,
        description=(
            "Enable device graph capture and replay for graph execution."
        ),
    )

    force: bool = Field(
        default=False,
        description=(
            "Skip validation of user provided flags against the architecture's "
            "required arguments."
        ),
    )

    kvcache_ce_watermark: float = Field(
        default=0.95,
        description=(
            "Projected cache usage threshold for scheduling CE requests."
        ),
    )

    decode_stall_timeout_s: float | None = Field(
        default=float(os.environ["MODULAR_DECODE_STALL_TIMEOUT_S"])
        if "MODULAR_DECODE_STALL_TIMEOUT_S" in os.environ
        else None,
        description=(
            "Seconds of no-batch-activity after which the decode worker exits."
        ),
    )

    decode_request_ttl_s: float | None = Field(
        default=float(os.environ["MODULAR_DECODE_REQUEST_TTL_S"])
        if "MODULAR_DECODE_REQUEST_TTL_S" in os.environ
        else None,
        description=("Per-request TTL in seconds for the decode-side dicts."),
    )

    enable_overlap_scheduler: bool = Field(
        default=False,
        description=("Whether to enable the overlap scheduler."),
    )

    dp_ce_balance_timeout_ms: float = Field(
        default=-1.0,
        description=(
            "Max deferral time in milliseconds for token-balanced CE "
            "scheduling across DP replicas. -1 disables the balancer; 0 "
            "places eagerly with post-cache weights; > 0 defers unbalanced "
            "CE work up to the deadline."
        ),
    )

    dp_ce_balance_threshold: float = Field(
        default=0.8,
        description=(
            "Per-step CE occupancy across DP replicas (0-1) at or above "
            "which CE work is scheduled without further deferral."
        ),
    )

    allow_unsupported_logprobs: bool = Field(
        default=False,
        description=(
            "When ``True``, requests that ask for ``logprobs`` against a "
            "runtime configuration that cannot honor them will raise a warning."
        ),
    )

    allow_extra_request_fields: bool = Field(
        default=False,
        description=(
            "When ``True``, unknown top-level fields on OpenAI-compatible "
            "request bodies are dropped with a warning."
        ),
    )

    prefer_module_v3: bool = Field(
        default=False,
        description=(
            "Whether to prefer the eager API architecture over the graph API architecture."
        ),
    )

    reasoning_parser: str | None = Field(
        default=None,
        description=("Name of the reasoning output parser."),
    )

    tool_parser: str | None = Field(
        default=None,
        description=("Name of the tool call parser."),
    )

    emit_reasoning_content: bool = Field(
        default=False,
        description=(
            "When ``True``, chat completion responses emit a thinking model's "
            "chain-of-thought under ``reasoning_content`` only."
        ),
    )

    temperature: float | None = Field(
        default=None,
        description=("Default sampling temperature."),
    )

    thinking_temperature: float | None = Field(
        default=None,
        description=(
            "Default temperature override for tokens inside ``<think>...</think>`` blocks."
        ),
    )

    max_vision_cache_entries: int = Field(
        default=256,
        description=(
            "Maximum number of images cached in the vision encoder cache."
        ),
    )

    denoising_cache: DenoisingCacheConfig = Field(
        default_factory=DenoisingCacheConfig,
        description=("Cache configuration for diffusion model denoising."),
    )

    # ------------------------------------------------------------------ #
    # Fields from SamplingConfig
    # ------------------------------------------------------------------ #

    in_dtype: DType = Field(
        default=DType.float32,
        description="The data type of the input tokens.",
    )

    out_dtype: DType = Field(
        default=DType.float32,
        description="The data type of the output logits.",
    )

    enable_structured_output: bool = Field(
        default=False,
        description=(
            "Enable structured generation/guided decoding for the server."
        ),
    )

    structured_output_backend: str | None = Field(
        default=None,
        description=(
            "Grammar backend for constrained decoding. One of ``xgrammar`` or "
            "``llguidance``. When unset (``None``), resolved during "
            "``PipelineConfig.resolve()`` to the architecture's default if it "
            "declares one, else the global default ``xgrammar``. An explicit "
            "value always wins."
        ),
    )

    enable_variable_logits: bool = Field(
        default=False,
        description=(
            "Enable the sampling graph to accept a ragged tensor of different sequences."
        ),
    )

    enable_penalties: bool = Field(
        default=False,
        description=(
            "Whether to apply frequency and presence penalties to the model's output."
        ),
    )

    enable_min_tokens: bool = Field(
        default=False,
        description=("Whether to enable ``min_tokens``."),
    )

    sample_on_host: bool = Field(
        default=False,
        description=(
            "Run the token sampler on the host CPU instead of the model device."
        ),
    )

    # ------------------------------------------------------------------ #
    # Fields from ProfilingConfig
    # ------------------------------------------------------------------ #

    gpu_profiling: GPUProfilingMode = Field(
        default="off",
        description="Whether to enable GPU profiling of the model.",
    )

    profiling_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the libkineto-backed HTA/Dynolog profiler."
        ),
    )

    profiling_output_path: str | None = Field(
        default=None,
        description=("Where to write the Chrome-trace JSON."),
    )

    profiling_dynolog_enabled: bool = Field(
        default=True,
        description=(
            "Whether to listen for Dynolog IPC on-demand-profile requests."
        ),
    )

    profiling_warmup_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of Model::execute() iterations to skip before recording."
        ),
    )

    profiling_active_steps: int = Field(
        default=10,
        ge=1,
        description=("Number of Model::execute() iterations to record."),
    )

    profiling_periodic_flush_seconds: int = Field(
        default=60,
        ge=1,
        description=(
            "Periodically flush in-flight trace chunks to disk every N seconds."
        ),
    )

    # ------------------------------------------------------------------ #
    # Sub-config objects (kept cohesive)
    # ------------------------------------------------------------------ #

    lora: LoRAConfig | None = Field(
        default=None,
        description="The LoRA config.",
    )

    speculative: SpeculativeConfig | None = Field(
        default=None,
        description="The SpeculativeConfig.",
    )

    draft_model: MAXModelConfig | None = Field(
        default=None,
        description=(
            "Draft model configuration for speculative decoding. "
            "Replaces the ``models['draft']`` entry in a :class:`PipelineConfig`."
        ),
    )

    # Escape hatch for multi-component pipelines (e.g. diffusion) where
    # a pre-built ModelManifest is required. When set,
    # PipelineConfig.from_args() uses this manifest directly instead of
    # constructing one from flat fields.
    _manifest_override: ModelManifest | None = PrivateAttr(default=None)

    # Cross-repo weight source (e.g. a bartowski GGUF repo supplying weights
    # for a meta-llama config repo). Not a user-settable input field -- set
    # directly on the instance (``args._weights_repo_id = ...``) by callers
    # that need it, then re-seeded onto the built MAXModelConfig by
    # MAXModelConfig.from_pipeline_args(), since that returns a fresh object
    # each call.
    _weights_repo_id: str | None = PrivateAttr(default=None)

    def __init__(
        self, *, models: ModelManifest | None = None, **data: Any
    ) -> None:
        super().__init__(**data)
        if models is not None:
            object.__setattr__(self, "_manifest_override", models)

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def main_architecture_name(self) -> str:
        """Returns the HuggingFace architecture class name for the main model.

        Reads ``architectures[0]`` from the model's HuggingFace config without
        constructing a full :class:`PipelineConfig`.

        Raises:
            ValueError: If the architecture name cannot be determined.
        """
        if self._manifest_override is not None:
            return self._manifest_override.main_architecture_name
        arch = MAXModelConfig.from_pipeline_args(self).architecture_name
        if arch is None:
            raise ValueError(
                f"Cannot determine architecture name for {self.model_path!r}: "
                "HuggingFace config has no 'architectures' field."
            )
        return arch

    @classmethod
    def from_flat_kwargs(cls, **kwargs: Any) -> Self:
        """Construct a :class:`PipelineArgs` from a flat CLI kwargs namespace.

        Routes flat kwargs (the same format accepted by
        :meth:`PipelineConfig.from_flat_kwargs`) into the appropriate flat
        fields of :class:`PipelineArgs`. Delegates to
        :meth:`PipelineConfig.from_flat_kwargs` for the heavy-lifting of
        routing and sub-config construction, then extracts the user-facing
        fields.

        Args:
            **kwargs: Flat keyword arguments, e.g. ``model_path``,
                ``kv_cache_size``, ``enable_lora``.

        Returns:
            A :class:`PipelineArgs` populated from the flat kwargs.
        """
        from max.pipelines.lib.config.config import PipelineConfig

        pipeline_config = PipelineConfig.from_flat_kwargs(**kwargs)
        return cls.from_pipeline_config(pipeline_config)

    @classmethod
    def from_pipeline_config(cls, pipeline_config: PipelineConfig) -> Self:
        """Construct a :class:`PipelineArgs` from an existing :class:`PipelineConfig`.

        Extracts the user-facing flat fields from a :class:`PipelineConfig`
        and returns a :class:`PipelineArgs` populated from them.

        This exists to let :meth:`from_flat_kwargs` reuse
        :meth:`PipelineConfig.from_flat_kwargs`'s flat-kwarg routing logic
        (parsing ``--model-override``, building the draft model config,
        etc.) instead of duplicating it. It is not a general round-trip:
        ``pipeline_config`` is expected to be freshly constructed and not
        yet resolved. Resolution-derived state (e.g. an applied dtype cast
        recorded by ``MAXModelConfig.resolve()``) is *not* preserved --
        :class:`PipelineArgs` is deliberately isolated from resolution
        mutations (see #90128), so passing an already-resolved
        ``pipeline_config`` here will silently drop that state.

        Args:
            pipeline_config: The source :class:`PipelineConfig` to extract
                from. Should not have had :meth:`PipelineConfig.resolve`
                called on it.

        Returns:
            A :class:`PipelineArgs` populated from the given config.
        """
        main = pipeline_config.models.get("main") or MAXModelConfig()
        # Multi-component (diffusion) manifests have no "main" entry; their
        # per-component configs can't be reconstructed from the flat fields,
        # so carry the manifest through verbatim.
        manifest = (
            pipeline_config.models
            if "main" not in pipeline_config.models
            else None
        )
        runtime = pipeline_config.runtime
        sampling = pipeline_config.sampling
        profiling = pipeline_config.profiling

        return cls(
            models=manifest,
            # top-level
            model_override=list(pipeline_config.model_override),
            task=pipeline_config.task,
            debug_verify_replay=pipeline_config.debug_verify_replay,
            # MAXModelConfig fields
            model_path=main.model_path,
            served_model_name=main.served_model_name,
            weight_path=list(main.weight_path),
            quantization_encoding=main.quantization_encoding,
            huggingface_model_revision=main.huggingface_model_revision,
            huggingface_weight_revision=main.huggingface_weight_revision,
            trust_remote_code=main.trust_remote_code,
            subfolder=main.subfolder,
            device_specs=list(main.device_specs),
            force_download=main.force_download,
            vision_config_overrides=dict(main.vision_config_overrides),
            rope_type=main.rope_type,
            sliding_window=main.sliding_window,
            enable_echo=main.enable_echo,
            chat_template=main.chat_template,
            use_subgraphs=main.use_subgraphs,
            data_parallel_degree=main.data_parallel_degree,
            pool_embeddings=main.pool_embeddings,
            max_length=main.max_length,
            kv_cache=main.kv_cache.model_copy(deep=True),
            # PipelineRuntimeConfig fields
            pipeline_role=runtime.pipeline_role,
            max_batch_size=runtime.max_batch_size,
            max_queue_size_tg=runtime.max_queue_size_tg,
            min_batch_size_tg=runtime.min_batch_size_tg,
            ep_size=runtime.ep_size,
            ep_use_allreduce=runtime.ep_use_allreduce,
            eplb_profile=runtime.eplb_profile,
            ce_delay_ms=runtime.ce_delay_ms,
            enable_prioritize_first_decode=runtime.enable_prioritize_first_decode,
            enable_chunked_prefill=runtime.enable_chunked_prefill,
            enable_in_flight_batching=runtime.enable_in_flight_batching,
            eplb_replicas_per_gpu=runtime.eplb_replicas_per_gpu,
            max_num_steps=runtime.max_num_steps,
            max_batch_input_tokens=runtime.max_batch_input_tokens,
            use_experimental_kernels=runtime.use_experimental_kernels,
            use_vendor_blas=runtime.use_vendor_blas,
            use_vendor_ccl=runtime.use_vendor_ccl,
            custom_architectures=list(runtime.custom_architectures),
            execute_empty_batches=runtime.execute_empty_batches,
            max_batch_total_tokens=runtime.max_batch_total_tokens,
            device_graph_capture=runtime.device_graph_capture,
            force=runtime.force,
            kvcache_ce_watermark=runtime.kvcache_ce_watermark,
            decode_stall_timeout_s=runtime.decode_stall_timeout_s,
            decode_request_ttl_s=runtime.decode_request_ttl_s,
            enable_overlap_scheduler=runtime.enable_overlap_scheduler,
            dp_ce_balance_timeout_ms=runtime.dp_ce_balance_timeout_ms,
            dp_ce_balance_threshold=runtime.dp_ce_balance_threshold,
            allow_unsupported_logprobs=runtime.allow_unsupported_logprobs,
            allow_extra_request_fields=runtime.allow_extra_request_fields,
            prefer_module_v3=runtime.prefer_module_v3,
            reasoning_parser=runtime.reasoning_parser,
            tool_parser=runtime.tool_parser,
            emit_reasoning_content=runtime.emit_reasoning_content,
            temperature=runtime.temperature,
            thinking_temperature=runtime.thinking_temperature,
            max_vision_cache_entries=runtime.max_vision_cache_entries,
            denoising_cache=runtime.denoising_cache.model_copy(deep=True),
            # SamplingConfig fields
            in_dtype=sampling.in_dtype,
            out_dtype=sampling.out_dtype,
            enable_structured_output=sampling.enable_structured_output,
            structured_output_backend=sampling.structured_output_backend,
            enable_variable_logits=sampling.enable_variable_logits,
            enable_penalties=sampling.enable_penalties,
            enable_min_tokens=sampling.enable_min_tokens,
            sample_on_host=sampling.sample_on_host,
            # ProfilingConfig fields
            gpu_profiling=profiling.gpu_profiling,
            profiling_enabled=profiling.profiling_enabled,
            profiling_output_path=profiling.profiling_output_path,
            profiling_dynolog_enabled=profiling.profiling_dynolog_enabled,
            profiling_warmup_steps=profiling.profiling_warmup_steps,
            profiling_active_steps=profiling.profiling_active_steps,
            profiling_periodic_flush_seconds=profiling.profiling_periodic_flush_seconds,
            # sub-configs
            lora=pipeline_config.lora.model_copy(deep=True)
            if pipeline_config.lora
            else None,
            speculative=pipeline_config.speculative.model_copy(deep=True)
            if pipeline_config.speculative
            else None,
            draft_model=pipeline_config.draft_model.model_copy(deep=True)
            if pipeline_config.draft_model is not None
            else None,
        )
