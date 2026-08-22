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

"""Standardized configuration for Pipeline Inference."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

from max.config import ConfigFileModel
from max.driver import accelerator_api
from max.engine import InferenceSession
from max.nn.comm import Signals
from max.pipelines.lib.arch_lookup import (
    find_architecture,
    import_custom_architectures,
)
from max.pipelines.lib.interfaces import (
    ArchConfig,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import (
    DISABLE_PARSER_SENTINEL,
    PipelineRuntimeConfig,
)
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.sampling import (
    DEFAULT_STRUCTURED_OUTPUT_ANY_WHITESPACE,
    DEFAULT_STRUCTURED_OUTPUT_BACKEND,
    SamplingConfig,
)
from max.pipelines.speculative.config import SpeculativeConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .model_config import (
    MAXModelConfig,
    _parse_component_overrides,
    _populate_weights_and_encoding,
    _select_quantization_encoding,
)
from .profiling_config import ProfilingConfig

logger = logging.getLogger("max.pipelines")

# ModelManifest is a dict[str, MAXModelConfig] subclass with extra methods.
# cyclopts (CLI framework) only recognizes plain dict types via typing.get_origin(),
# which returns None for concrete subclasses. At runtime, Pydantic sees
# dict[str, MAXModelConfig] so cyclopts can resolve CLI paths like
# --pipeline.models.main.model-path. mypy sees ModelManifest so methods like
# .with_override(), .resolve(), .main_architecture_name type-check correctly.
if TYPE_CHECKING:
    from max.pipelines.lib.pipeline_args import PipelineArgs

    _ModelsType = ModelManifest
else:
    _ModelsType = dict[str, MAXModelConfig]


def _nested_model_class(annotation: Any) -> type[BaseModel] | None:
    """Return the Pydantic model class for a field annotation, if any.

    Unwraps ``Optional``/``Union`` annotations (e.g. ``KVCacheConfig | None``)
    and returns the first :class:`~pydantic.BaseModel` subclass found. Returns
    ``None`` for non-model annotations such as ``dict[str, Any]`` or ``str``,
    so plain data dicts are never treated as nested config sub-models.
    """
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


_SubConfigT = TypeVar("_SubConfigT", bound=ConfigFileModel)


def _construct_from_user_fields(sub: _SubConfigT) -> _SubConfigT:
    """Constructs a fresh sub-config from only the caller-set fields.

    The result is built once, through the class constructor: unset fields
    re-derive from the class defaults and nested models are rebuilt rather
    than aliased, so it shares no mutable state with ``sub``.
    """
    return type(sub)(
        **sub.model_dump(
            include=sub.model_fields_set - {"config_file", "section_name"}
        )
    )


def _is_disable_parser_sentinel(value: str | None) -> bool:
    """Return ``True`` if ``value`` is the case-insensitive disable sentinel.

    Users can pass the string ``"none"`` (case-insensitive) to
    ``runtime.reasoning_parser`` or ``runtime.tool_parser`` to explicitly
    disable the parser, overriding any architecture-declared default.
    """
    return isinstance(value, str) and value.lower() == DISABLE_PARSER_SENTINEL


class PipelineConfig(ConfigFileModel):
    """Configuration for a pipeline.

    Contains settings for model selection, batch sizing, sampling, profiling,
    LoRA adapters, and speculative decoding. Once initialized, all fields are
    resolved to their final values from CLI flags, config files, environment
    variables, or internal defaults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    debug_verify_replay: bool = Field(
        default=False,
        description=(
            "When ``device_graph_capture`` is enabled, execute eager launch-trace "
            "verification before replay. Intended for debugging only."
        ),
    )
    """Whether to run eager verification before device graph replay."""

    models: _ModelsType = Field(
        default_factory=ModelManifest,
        description="The model manifest containing all model configs keyed by role.",
    )
    """The model manifest containing all model configs keyed by role."""

    model_override: list[str] = Field(
        default_factory=list,
        description=(
            "Per-component overrides for the ModelManifest, in the format "
            "``component.field=value``. Applied before resolution. Repeatable. "
            "Example: ``transformer.quantization_encoding=float4_e2m1fnx2``."
        ),
    )
    """Per-component model overrides applied before resolution."""

    @staticmethod
    def _normalize_models_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize dash-keyed dicts from cyclopts CLI parsing to underscores.

        When cyclopts parses CLI args like ``--pipeline.models.main.model-path``,
        it produces nested dicts with dash-separated keys (e.g.
        ``{"main": {"model-path": "value"}}``).  Pydantic expects underscore-
        separated field names, so we normalise before validation.

        Normalization recurses into nested config sub-models so fields like
        ``--pipeline.models.main.kv-cache.kv-cache-format`` resolve to
        ``{"main": {"kv_cache": {"kv_cache_format": ...}}}``. Recursion is
        schema-aware: it only descends into keys whose field is a Pydantic
        model, leaving plain data dicts (e.g. ``vision_config_overrides``)
        untouched so legitimately-dashed data keys are preserved.

        Raises:
            ValueError: If two keys at the same level normalize to the same
                field name (e.g. mixing ``kv-cache`` and ``kv_cache``), which
                would otherwise silently drop one of the values.
        """

        def normalize(
            raw: dict[str, Any], model_cls: type[BaseModel]
        ) -> dict[str, Any]:
            fields = model_cls.model_fields
            normalized: dict[str, Any] = {}
            for key, value in raw.items():
                norm_key = key.replace("-", "_")
                if norm_key in normalized:
                    raise ValueError(
                        f"Conflicting CLI keys normalize to '{norm_key}' on "
                        f"{model_cls.__name__}: '{key}' collides with an "
                        "earlier key. Use one consistent spelling (e.g. "
                        f"'--...{norm_key.replace('_', '-')}'), not a mix of "
                        "dashes and underscores."
                    )
                field = fields.get(norm_key)
                nested_cls = (
                    _nested_model_class(field.annotation) if field else None
                )
                if nested_cls is not None and isinstance(value, dict):
                    normalized[norm_key] = normalize(value, nested_cls)
                else:
                    normalized[norm_key] = value
            return normalized

        result: dict[str, Any] = {}
        for role, value in data.items():
            result[role] = (
                normalize(value, MAXModelConfig)
                if isinstance(value, dict)
                else value
            )
        return result

    @model_validator(mode="before")
    @classmethod
    def _drop_unrequested_optional_subtrees(cls, data: Any) -> Any:
        """Drop optional subtrees whose enabling field wasn't supplied.

        The CLI generates a default for every flag, so a subtree's presence
        cannot signal intent -- only its enabling field can.
        """
        if not isinstance(data, dict):
            return data
        for subtree, enabling_field in (
            ("lora", "enable_lora"),
            ("speculative", "speculative_method"),
        ):
            section = data.get(subtree)
            if isinstance(section, dict) and not section.get(enabling_field):
                data = {k: v for k, v in data.items() if k != subtree}
        return data

    @field_validator("models", mode="wrap")
    @classmethod
    def _coerce_models(cls, v: Any, handler: Any) -> ModelManifest:
        if isinstance(v, ModelManifest):
            return v
        if isinstance(v, dict):
            v = cls._normalize_models_dict(v)
        result = handler(v)
        if isinstance(result, ModelManifest):
            return result
        return ModelManifest(result)

    @model_validator(mode="before")
    @classmethod
    def _disable_penalties_with_draft_model(cls, data: Any) -> Any:
        """Force penalties off when speculative decoding is configured.

        The speculative-decoding pipelines don't support the penalty
        sampling features. Both facts are user input, so the override
        edits the raw constructor input; the constructed config is
        never mutated.
        """
        if not isinstance(data, dict):
            return data
        models = data.get("models")
        if not (isinstance(models, dict) and "draft" in models):
            return data
        raw_sampling = data.get("sampling")
        if raw_sampling is None:
            return data
        if isinstance(raw_sampling, SamplingConfig):
            sampling = raw_sampling
        else:
            try:
                sampling = SamplingConfig.model_validate(raw_sampling)
            except ValidationError:
                # Malformed input: fall through so field validation
                # reports the error with proper field locations.
                return data
        if not sampling.enable_penalties:
            return data
        logger.warning(
            "frequency_penalty, presence_penalty and repetition_penalty are not currently supported with speculative decoding."
        )
        return {
            **data,
            "sampling": sampling.model_copy(update={"enable_penalties": False}),
        }

    @model_validator(mode="after")
    def _validate_lora_prefix_caching(self) -> Self:
        """Reject LoRA combined with prefix caching.

        Both settings are user input, so the check runs at construction
        rather than ``resolve()``.
        """
        if (
            self.lora
            and self.lora.enable_lora
            and "main" in self.models
            and self.models["main"].kv_cache.enable_prefix_caching
        ):
            raise ValueError(
                "LoRA is not compatible with prefix caching. "
                "Please disable prefix caching by using the "
                "--no-enable-prefix-caching flag."
            )
        return self

    @property
    def model(self) -> MAXModelConfig:
        """The main model config. Alias for ``models["main"]``."""
        main = self.models.get("main")
        if main is None:
            raise ValueError(
                "No main model configured. For diffusion pipelines, access "
                "component models via pipeline_config.models[<role>]."
            )
        return main

    @property
    def draft_model(self) -> MAXModelConfig | None:
        """The draft model configuration. Alias for ``models.get("draft")``."""
        return self.models.get("draft")

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, description="The sampling config."
    )
    """The sampling configuration."""

    profiling: ProfilingConfig = Field(
        default_factory=ProfilingConfig, description="The profiling config."
    )
    """The profiling configuration."""

    lora: LoRAConfig | None = Field(
        default=None, description="The LoRA config."
    )
    """The LoRA configuration."""

    speculative: SpeculativeConfig | None = Field(
        default=None, description="The SpeculativeConfig."
    )
    """The speculative decoding configuration."""

    runtime: PipelineRuntimeConfig = Field(
        default_factory=PipelineRuntimeConfig,
        description="Model-agnostic runtime settings for pipeline execution.",
    )
    """The model-agnostic runtime settings for pipeline execution."""

    task: PipelineTask = Field(
        default=PipelineTask.UNDEFINED,
        description=(
            "The pipeline task to run (e.g. ``text_generation``, "
            "``embeddings_generation``). Used to disambiguate architectures "
            "registered under the same name for multiple tasks."
        ),
    )
    """The pipeline task, used for arch disambiguation during config resolution."""

    @property
    def needs_bitmask_constraints(self) -> bool:
        """Whether constrained decoding can fire and requires the bitmask path.

        True if the user enabled ``--enable-structured-output`` (for
        user-supplied ``response_format=json_schema``) or a tool parser is
        configured (tool-call grammars work without the flag — they are
        server-generated and gated on having a parser that can both produce
        the grammar and parse the resulting output).

        Tool-call constrained decoding can be turned off independently via
        ``sampling.enable_tool_call_constrained_decode``: when that is
        ``False`` the tool parser still parses tool calls out of generated
        text, but no grammar is generated and the bitmask path is not needed
        on its account.

        Drives whether model / sampler graphs are compiled with a bitmask
        input and whether the D2H pinned buffer is allocated. Distinct from
        ``sampling.enable_structured_output``, which is the user-facing
        flag and only gates honoring user-supplied JSON schemas.
        """
        return self.sampling.enable_structured_output or (
            self.runtime.tool_parser is not None
            and self.sampling.enable_tool_call_constrained_decode
        )

    _config_file_section_name: str = PrivateAttr(default="pipeline_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    def configure_session(self, session: InferenceSession) -> None:
        """Configures a :class:`~max.engine.InferenceSession` with standard pipeline settings."""
        session.gpu_profiling(self.profiling.gpu_profiling)
        session._use_experimental_kernels(self.runtime.use_experimental_kernels)
        session._use_vendor_blas(self.runtime.use_vendor_blas)
        session._use_vendor_ccl(self.runtime.use_vendor_ccl)
        # BLASST prefill sparsity sweep hook (off unless ENABLE_BLASST is set in
        # the env). Injects the comptime defines the SM100 2Q attention kernel
        # reads via get_defined_{bool,int}. Values must be str/int, never Python
        # True (a UnitAttr fails KGEN's string coercion). No effect when unset.
        if os.environ.get("ENABLE_BLASST"):
            session._set_mojo_define("ENABLE_BLASST", "true")
            session._set_mojo_define(
                "BLASST_LOG_THRESHOLD_MAG",
                int(os.environ.get("BLASST_LOG_THRESHOLD_MAG", "13000")),
            )

    def estimate_signal_buffer_memory(
        self, arch_config: ArchConfig | None = None
    ) -> int:
        """Estimates total signal-buffer memory across all devices.

        Signal buffers are fixed-size (:attr:`~max.nn.comm.allreduce.Signals.NUM_BYTES`)
        per-GPU allocations used by P2P collectives. The only site visible
        from :class:`PipelineConfig` is the main model graph, and only for
        multi-GPU pipelines. The ``tiered``/``rust_tiered`` KV connectors fan
        MLA-replicated blocks out via plain P2P copies, not a signal-buffer
        broadcast (see ``dkv/kv-tier-connector/src/copy_engine.rs``), so they
        contribute no additional term here.

        Returns 0 for single-device pipelines.

        Args:
            arch_config: Unused; kept for interface parity with
                :meth:`MemoryPlanner.estimate_signal_buffer_memory`.

        Returns:
            Estimated total signal-buffer memory in bytes (across all devices).
        """
        ngpus = len(self.model.device_specs)
        if ngpus <= 1:
            return 0
        return Signals.NUM_BYTES * ngpus

    def _apply_speculative_draft_architecture(self) -> None:
        """Rewrite the draft model's HuggingFace architecture for the method.

        Runs after the models are built, since it edits the draft's loaded
        HuggingFace config rather than any MAX config field.
        """
        if self.speculative is None:
            return
        # We need to set the architecture to LlamaForCausalLMEagle for Eagle speculative decoding
        if self.speculative.is_eagle() and self.draft_model is not None:
            if len(self.draft_model.huggingface_config.architectures) != 1:
                raise ValueError(
                    f"Expected exactly 1 architecture in draft model config, "
                    f"got {len(self.draft_model.huggingface_config.architectures)}"
                )
            hf_arch = self.draft_model.huggingface_config.architectures[0]
            if hf_arch == "LlamaForCausalLM":
                self.draft_model.huggingface_config.architectures[0] = (
                    "LlamaForCausalLMEagle"
                )
        # DFlash drafts ship with architectures: ["DFlashDraftModel"],
        # which isn't registered as a standalone MAX architecture (the draft
        # is only ever invoked through UnifiedDflashLlama3). Override to
        # LlamaForCausalLM.
        if self.speculative.is_dflash() and self.draft_model is not None:
            if len(self.draft_model.huggingface_config.architectures) != 1:
                raise ValueError(
                    f"Expected exactly 1 architecture in draft model config, "
                    f"got {len(self.draft_model.huggingface_config.architectures)}"
                )
            hf_arch = self.draft_model.huggingface_config.architectures[0]
            if hf_arch == "DFlashDraftModel":
                self.draft_model.huggingface_config.architectures[0] = (
                    "LlamaForCausalLM"
                )

    def _validate_repo_access(self) -> None:
        """Validates that every model's repo was provided and is accessible.

        Called at the end of the construction factories so a bad repo fails
        fast. See :meth:`MAXModelConfig.validate_repo_access`.
        """
        for model in self.models.values():
            model.validate_repo_access()

    def _validate_required_arguments_against_architecture(
        self, architecture: Any
    ) -> None:
        """Validates and overrides config from architecture required_arguments.

        Checks the required_arguments dictionary from the architecture
        and automatically overrides any config values that don't match, logging warnings
        when changes are made.

        Args:
            architecture: The SupportedArchitecture containing required_arguments dictionary
        """
        if not architecture.required_arguments:
            return

        config_objects = [
            ("PipelineConfig", self),
            ("PipelineRuntimeConfig", self.runtime),
            ("MAXModelConfig", self.model),
            ("SamplingConfig", self.sampling),
            ("KVCacheConfig", self.model.kv_cache),
        ]

        # Add draft model configurations if present
        if self.draft_model is not None:
            config_objects.extend(
                [
                    ("Draft_MAXModelConfig", self.draft_model),
                    (
                        "Draft_KVCacheConfig",
                        self.draft_model.kv_cache,
                    ),
                ]
            )

        for arg_name, required_value in architecture.required_arguments.items():
            # Check each config object for the required argument
            for config_name, config_obj in config_objects:
                current_value = getattr(config_obj, arg_name, required_value)
                if current_value != required_value:
                    logger.warning(
                        f"Architecture '{architecture.name}' requires {config_name}.{arg_name}={required_value}, "
                        f"overriding current value {current_value}"
                    )
                    setattr(config_obj, arg_name, required_value)
                # We should be able to override this value for all config objects.
                continue

    def _apply_speculative_target_architecture(self) -> None:
        """Override the target architecture for unified spec-decode pipelines.

        Unified EAGLE / DFlash / MTP pipelines fold the draft into a dedicated
        target architecture (e.g. ``DeepseekV3ForCausalLM`` →
        ``UnifiedMTPDeepseekV3ForCausalLM``). This mutates
        ``model.huggingface_config.architectures[0]`` in place.

        This must run *before* the architecture is resolved from
        ``models.main_architecture_name``, so that the resolved ``arch`` —
        consumed by memory estimation, the overlap scheduler, parser
        resolution, and ``pipeline_model`` construction — reflects the
        override. ``from_args`` invokes it before construction-time
        resolution. It is a no-op when speculative decoding is disabled.
        """
        if not self.speculative:
            return

        target_archs = self.model.huggingface_config.architectures
        if target_archs[0] == "LlamaForCausalLM":
            if self.speculative.is_dflash():
                target_archs[0] = "UnifiedDflashLlama3ForCausalLM"
            else:
                target_archs[0] = "UnifiedEagleLlama3ForCausalLM"
        if target_archs[0] == "DeepseekV3ForCausalLM":
            # Choose between MTP (NextN layer baked into target ckpt) and
            # Eagle3 (separate draft ckpt with arch
            # ``Eagle3DeepseekV2ForCausalLM``) based on the draft arch.
            draft_archs = (
                self.draft_model.huggingface_config.architectures
                if self.draft_model is not None
                else None
            )
            if draft_archs is None:
                target_archs[0] = "UnifiedMTPDeepseekV3ForCausalLM"
            elif (
                draft_archs and draft_archs[0] == "Eagle3DeepseekV2ForCausalLM"
            ):
                target_archs[0] = "Eagle3DeepseekV3ForCausalLM"
            elif draft_archs and draft_archs[0] == "LlamaForCausalLMEagle3":
                target_archs[0] = "Eagle3MHADeepseekV3ForCausalLM"
            else:
                if not draft_archs:
                    raise ValueError(
                        "Draft model HF config has empty"
                        " ``architectures=[]``. Expected"
                        " 'Eagle3DeepseekV2ForCausalLM' (Eagle3 draft),"
                        " 'LlamaForCausalLMEagle3' (Llama MHA Eagle3"
                        " draft), or no draft model (MTP path)."
                    )
                raise ValueError(
                    "Unrecognized draft architecture for DeepseekV3"
                    f" target: {draft_archs[0]!r}. Expected"
                    " 'Eagle3DeepseekV2ForCausalLM' (Eagle3 draft),"
                    " 'LlamaForCausalLMEagle3' (Llama MHA Eagle3 draft),"
                    " or no draft model (MTP path)."
                )
        if target_archs[0] == "KimiK25ForConditionalGeneration":
            draft_archs = (
                self.draft_model.huggingface_config.architectures
                if self.draft_model is not None
                else None
            )
            if self.speculative.is_dflash():
                target_archs[0] = "UnifiedDflashKimiK25ForCausalLM"
            elif draft_archs and draft_archs[0] == "LlamaForCausalLMEagle3":
                # MLA target + MHA (Llama-style) Eagle3 draft.
                target_archs[0] = "Eagle3MHAKimiK25ForCausalLM"
            else:
                # MLA target + MLA Eagle3 draft (existing path).
                target_archs[0] = "Eagle3DeepseekV2ForCausalLM"
        if target_archs[0] == "Gemma4ForConditionalGeneration":
            draft_archs = (
                self.draft_model.huggingface_config.architectures
                if self.draft_model is not None
                else None
            )
            if draft_archs and draft_archs[0] == "Gemma4AssistantForCausalLM":
                target_archs[0] = "UnifiedMTPGemma4ForCausalLM"
            elif draft_archs and draft_archs[0] == "DSparkDraftModel":
                # Speculators-format DSpark drafters (e.g.
                # RedHatAI/gemma-4-31B-it-speculator.dspark) declare the
                # generic architectures: ["DSparkDraftModel"].
                target_archs[0] = "UnifiedDSparkGemma4_31BForCausalLM"
            elif (
                self.speculative.is_dflash()
                and draft_archs
                # z-lab DFlash drafters (e.g. z-lab/gemma-4-31B-it-DFlash)
                # declare architectures: ["DFlashDraftModel"], which
                # ``_create_speculative_config_if_needed`` rewrites to
                # "LlamaForCausalLM" on the CLI-kwargs path (but not the
                # recipe path) before this runs. Accept both spellings.
                and draft_archs[0] in ("DFlashDraftModel", "LlamaForCausalLM")
            ):
                target_archs[0] = "UnifiedDflashGemma4_31BForCausalLM"
        # Gemma 4 12B ships as the "gemma4_unified" model line; its DSpark
        # block drafter declares architectures: ["Gemma4DSparkModel"].
        if target_archs[0] == "Gemma4UnifiedForConditionalGeneration":
            draft_archs = (
                self.draft_model.huggingface_config.architectures
                if self.draft_model is not None
                else None
            )
            if draft_archs and draft_archs[0] == "Gemma4DSparkModel":
                target_archs[0] = "UnifiedDSparkGemma4_12BForCausalLM"
        if target_archs[0] == "MiniMaxM3SparseForConditionalGeneration":
            draft_archs = (
                self.draft_model.huggingface_config.architectures
                if self.draft_model is not None
                else None
            )
            if self.speculative.is_mtp() and self.draft_model is None:
                target_archs[0] = (
                    "UnifiedMTPMiniMaxM3SparseForConditionalGeneration"
                )
            elif draft_archs and draft_archs[0] == "LlamaForCausalLMEagle3":
                # M3 target + MHA (Llama-style) Eagle3 draft. The v0 Eagle3
                # path forbids block-sparse attention.
                target_archs[0] = (
                    "Eagle3MHAMiniMaxM3SparseForConditionalGeneration"
                )
        if target_archs[0] == "Qwen3_5ForConditionalGeneration":
            # Qwen3.8 bakes a NextN MTP head into the target checkpoint, so
            # there is no separate draft model. Qwen3.5 shares the arch name
            # but ships no head; only override when the head exists. Unlike
            # the other in-checkpoint MTP overrides this one also waits for an
            # explicit `--speculative-method mtp`: the fused graph it selects
            # is served by Mach rather than MAX, so a plain `max serve` of a
            # Qwen3.8 checkpoint must keep landing on the base architecture.
            text_config = getattr(
                self.model.huggingface_config,
                "text_config",
                self.model.huggingface_config,
            )
            has_mtp = (
                getattr(text_config, "mtp_num_hidden_layers", 0) or 0
            ) > 0
            if (
                self.speculative.is_mtp()
                and self.draft_model is None
                and has_mtp
            ):
                target_archs[0] = "UnifiedMTPQwen3_5ForConditionalGeneration"
        if target_archs[0] == "GlmMoeDsaForCausalLM":
            # GLM-5.2 bakes a NextN MTP layer into the target checkpoint, so
            # there is no separate draft model. GLM-5.1 shares the arch name
            # but has no MTP layer; only override when MTP weights exist.
            has_mtp = (
                getattr(
                    self.model.huggingface_config,
                    "num_nextn_predict_layers",
                    0,
                )
                or 0
            ) > 0
            if self.draft_model is None and has_mtp:
                target_archs[0] = "UnifiedMTPGlmMoeDsaForCausalLM"
        if target_archs[0] == "InklingForConditionalGeneration":
            # Inkling bakes chained MTP depths into the target checkpoint.
            mtp_config = getattr(
                self.model.huggingface_config, "mtp_config", None
            )
            n_mtp = (
                mtp_config.get("num_nextn_predict_layers")
                if isinstance(mtp_config, dict)
                else getattr(mtp_config, "num_nextn_predict_layers", None)
            )
            if self.draft_model is None and (n_mtp or 0) > 0:
                target_archs[0] = "UnifiedMTPInklingForConditionalGeneration"

    def _validate_synthetic_acceptance_with_constrained_decoding(self) -> None:
        """Rejects synthetic acceptance when constrained decoding can fire.

        The synthetic acceptance path ignores token bitmasks, so structured
        output and tool-call grammars would silently stop being enforced
        while the serve layer believes they are. Checked here — after tool
        parser resolution — because both inputs to the decision
        (``needs_bitmask_constraints`` and the speculative section) are only
        final at this point, and checking per-model would leave every spec
        arch to re-implement it.
        """
        if self.speculative is None:
            return
        if self.speculative.synthetic_acceptance_rate is None:
            return
        if not self.needs_bitmask_constraints:
            return
        raise ValueError(
            "synthetic_acceptance_rate is incompatible with constrained"
            " decoding: the synthetic acceptance path ignores token"
            " bitmasks, so structured output and tool-call grammars would"
            " silently stop being enforced. For synthetic-acceptance"
            " benchmarking pass --tool-parser none and leave"
            " --enable-structured-output off."
        )

    def _resolve_default_reasoning_parser(self, arch: Any = None) -> None:
        """Apply the architecture's default reasoning parser when unset.

        If the user did not configure ``runtime.reasoning_parser`` and the
        resolved ``SupportedArchitecture`` declares a default
        ``reasoning_parser``, use it. Explicit user configuration always wins.

        Passing the case-insensitive sentinel ``"none"`` explicitly disables
        the reasoning parser; the value is normalized to ``None`` and the
        architecture default is skipped.
        """
        if _is_disable_parser_sentinel(self.runtime.reasoning_parser):
            self.runtime.reasoning_parser = None
            logger.info(
                "Reasoning parser explicitly disabled, skipping architecture default."
            )
            return

        if self.runtime.reasoning_parser is not None:
            return

        if arch is None or arch.reasoning_parser is None:
            return

        self.runtime.reasoning_parser = arch.reasoning_parser
        logger.info(
            "Defaulting reasoning parser to %r for architecture %s. "
            "Override with --reasoning-parser, or pass "
            "--reasoning-parser=none to disable.",
            arch.reasoning_parser,
            arch.name,
        )

    def _resolve_default_tool_parser(self, arch: Any = None) -> None:
        """Apply the architecture's default tool parser when unset.

        If the user did not configure ``runtime.tool_parser`` and the
        resolved ``SupportedArchitecture`` declares a default
        ``tool_parser``, use it. Explicit user configuration always wins.

        Passing the case-insensitive sentinel ``"none"`` explicitly disables
        the tool parser; the value is normalized to ``None`` and the
        architecture default is skipped.
        """
        if _is_disable_parser_sentinel(self.runtime.tool_parser):
            self.runtime.tool_parser = None
            logger.info(
                "Tool parser explicitly disabled, skipping architecture default.",
            )
            return

        if self.runtime.tool_parser is not None:
            return

        if arch is None or arch.tool_parser is None:
            return

        if callable(arch.tool_parser):
            parser_name = arch.tool_parser(self.model.huggingface_model_repo)
        else:
            parser_name = arch.tool_parser

        self.runtime.tool_parser = parser_name
        logger.info(
            "Defaulting tool parser to %r for architecture %s. "
            "Override with --tool-parser, or pass --tool-parser=none "
            "to disable.",
            parser_name,
            arch.name,
        )

    def _resolve_default_structured_output_backend(
        self, arch: Any = None
    ) -> None:
        """Resolve the structured output backend to a concrete value.

        Resolution order (highest precedence first):

        1. An explicit user choice (``sampling.structured_output_backend`` is
           not ``None``) always wins -- including an explicit ``"xgrammar"`` on
           an architecture that pins ``"llguidance"``.
        2. Otherwise, if the resolved ``SupportedArchitecture`` declares a
           ``default_structured_output_backend`` (e.g. Gemma 3 / MiniMax-M2 pin
           ``"llguidance"``), use it.
        3. Otherwise, fall back to the global default ``"xgrammar"``.

        Runs whenever construction resolves an architecture, so the field is
        a concrete ``str`` on any config with a registered architecture. The
        ``None`` sentinel (unset) is what distinguishes an explicit user
        value from the default -- mirroring the reasoning/tool parser
        resolvers above.
        """
        if self.sampling.structured_output_backend is not None:
            # Explicit user configuration always wins.
            return

        if (
            arch is not None
            and arch.default_structured_output_backend is not None
        ):
            self.sampling.structured_output_backend = (
                arch.default_structured_output_backend
            )
            logger.info(
                "Defaulting structured output backend to %r for architecture "
                "%s. Override with --structured-output-backend.",
                arch.default_structured_output_backend,
                arch.name,
            )
            return

        self.sampling.structured_output_backend = (
            DEFAULT_STRUCTURED_OUTPUT_BACKEND
        )
        logger.info(
            "Defaulting structured output backend to the global default %r "
            "(architecture %s declares no default). Override with "
            "--structured-output-backend.",
            DEFAULT_STRUCTURED_OUTPUT_BACKEND,
            arch.name if arch is not None else None,
        )

    def _resolve_default_structured_output_any_whitespace(
        self, arch: Any = None
    ) -> None:
        """Resolve structured-output whitespace mode based on architecture."""
        if self.sampling.structured_output_any_whitespace is not None:
            # Explicit user configuration always wins.
            return

        if (
            arch is not None
            and arch.default_structured_output_any_whitespace is not None
        ):
            self.sampling.structured_output_any_whitespace = (
                arch.default_structured_output_any_whitespace
            )
            logger.info(
                "Using architecture default structured output any_whitespace %r"
                " (%s).",
                arch.default_structured_output_any_whitespace,
                arch.name,
            )
            return

        self.sampling.structured_output_any_whitespace = (
            DEFAULT_STRUCTURED_OUTPUT_ANY_WHITESPACE
        )

    def _resolve_max_length(self, arch: Any, draft_arch: Any = None) -> None:
        """Resolves each model's ``max_length`` through its architecture.

        The architecture owns the rule — whether the checkpoint's length is a
        hard cap or just a default — and it runs here, once. Memory planning
        may lower the result to fit the device, but only on the plan.
        """
        # Capture intent before overwriting: anything that set max_length
        # since __init__ counts as user-provided.
        self.model._max_length_user_provided = self.model.max_length is not None
        self.model.max_length = arch.config.calculate_max_seq_len(
            self, self.model.huggingface_config, self.model
        )
        if draft_arch is not None and self.draft_model is not None:
            self.draft_model._max_length_user_provided = (
                self.draft_model.max_length is not None
            )
            self.draft_model.max_length = (
                draft_arch.config.calculate_max_seq_len(
                    self,
                    self.draft_model.huggingface_config,
                    self.draft_model,
                )
            )

    def _validate_and_resolve_overlap_scheduler(self, arch: Any = None) -> None:
        if not self.runtime.force:
            if (
                self.runtime.device_graph_capture is None
                and arch is not None
                and arch.supports_device_graph_capture
                and accelerator_api() in ("cuda", "hip")
                and self._is_eligible_for_overlap_serve_optimizations(arch)
                # Device graph capture is not supported for prefill-only workers.
                and self.runtime.pipeline_role != "prefill_only"
            ):
                self.runtime.device_graph_capture = True
                logger.info(
                    "Automatically enabling device graph capture for %s. "
                    "You can manually disable this by setting --no-device-graph-capture.",
                    arch.name,
                )

        if self.runtime.device_graph_capture is None:
            self.runtime.device_graph_capture = False

        self._validate_and_resolve_device_graph_capture()

        if self.runtime.force:
            return

        # Automatically enable overlap scheduling for architectures that declare
        # support. New architectures opt out by setting ``supports_overlap_scheduler=False``.
        if not self.runtime.enable_overlap_scheduler:
            if (
                arch is not None
                and arch.supports_overlap_scheduler
                and self._is_eligible_for_overlap_serve_optimizations(arch)
            ):
                self.runtime.enable_overlap_scheduler = True
                logger.info(
                    f"Automatically enabling overlap scheduling for {arch.name}. "
                    "You can manually disable this by setting --no-enable-overlap-scheduler --force."
                )

        # Raise errors when we detect features that are not compatible with the overlap scheduler.
        if self.runtime.enable_overlap_scheduler:
            if self.runtime.pipeline_role in ("decode_only", "prefill_only"):
                logger.info(
                    "Overlap scheduling enabled for %s worker "
                    "(Disaggregated Inference). THIS IS EXPERIMENTAL.",
                    self.runtime.pipeline_role,
                )
            if self.sampling.enable_variable_logits:
                raise ValueError(
                    "Variable logits are not supported with the Overlap scheduler. "
                )
            if self.lora:
                raise ValueError(
                    "LoRA is not supported with the Overlap scheduler."
                )
            if self.model.default_device_spec.device_type == "cpu":
                raise ValueError(
                    "Overlap scheduler is not supported with CPU models."
                )

    def _is_eligible_for_overlap_serve_optimizations(self, arch: Any) -> bool:
        # Overlap scheduling and device graph capture are only supported for
        # text generation. Auto-enabling them for other tasks (e.g. embeddings)
        # would fail downstream pipeline construction. See
        # `get_pipeline_for_task` in registry.py.
        return (
            arch.task == PipelineTask.TEXT_GENERATION
            and not self.sampling.enable_variable_logits
            and not self.lora
            and self.model.default_device_spec.device_type != "cpu"
        )

    def _validate_and_resolve_device_graph_capture(self) -> None:
        if not self.runtime.device_graph_capture:
            return

        if not self.runtime.enable_overlap_scheduler:
            logger.info("Enabling overlap scheduling for device graph capture.")
        self.runtime.enable_overlap_scheduler = True

    def _validate_pipeline_config_for_speculative_decoding(
        self,
        target_arch: Any,
        draft_arch: Any,
    ) -> None:
        """Validates pipeline config when used in speculative decoding mode.

        Args:
            target_arch: Pre-resolved target architecture from the registry.
            draft_arch: Pre-resolved draft architecture from the registry.
        """
        assert self.draft_model is not None
        assert self.speculative is not None

        if self.model.enable_echo:
            raise ValueError(
                "enable_echo not currently supported with speculative decoding enabled"
            )

    def _validate_model_config_against_arch(
        self, model_config: MAXModelConfig, arch: Any
    ) -> None:
        """Validates model config fields against a resolved architecture.

        Validates quantization encoding, LoRA support, multi-GPU
        compatibility, and empty-batch support. Read-only for encoding and
        weight paths — those are resolved at construction
        (:meth:`_populate_model_configs_from_archs`). Does not
        perform memory estimation.

        Args:
            model_config: The model configuration to validate.
            arch: The pre-resolved architecture to validate against.
        """
        # Validate that model supports empty batches, if being requested.
        if (
            self.runtime.execute_empty_batches
            and not arch.supports_empty_batches
        ):
            raise ValueError(
                f"Architecture '{arch.name}' does not support empty batches. "
                "Please set `execute_empty_batches` to False."
            )

        # Validate LoRA support - currently only Llama3 models support LoRA
        if self.lora and self.lora.enable_lora:
            # Check if the architecture is Llama3 (LlamaForCausalLM)
            if "LlamaForCausalLM" not in arch.name:
                raise ValueError(
                    f"LoRA is not currently supported for architecture '{arch.name}'. "
                    f"LoRA support is currently only available for Llama-3.x models (LlamaForCausalLM architecture). "
                    f"Model '{model_config.model_path}' uses the '{arch.name}' architecture."
                )
            # Currently, LoRA supported on only 1 device.
            if len(model_config.device_specs) > 1:
                raise ValueError(
                    "LoRA is currently not supported with the number of devices > 1."
                )

        model_config.validate_multi_gpu_supported(
            multi_gpu_supported=arch.multi_gpu_supported
        )

        # Re-check after the required-argument overrides, which may rewrite
        # the encoding populated earlier in construction.
        resolved_encoding = _select_quantization_encoding(
            model_config, arch.default_encoding
        )
        if resolved_encoding not in arch.supported_encodings:
            raise ValueError(
                f"quantization_encoding of '{resolved_encoding}' not supported by MAX engine."
            )

    def _validate_speculative_model_configs(
        self, target_arch: Any, draft_arch: Any
    ) -> None:
        """Validates model configs for unified speculative decoding.

        Args:
            target_arch: Pre-resolved target architecture from the registry.
            draft_arch: Pre-resolved draft architecture from the registry.
        """
        assert self.draft_model is not None

        # Note: quantization_encoding is NOT inherited from the target model.
        # Draft models (especially EAGLE3) typically use bfloat16 regardless
        # of the target model's quantization. The draft model auto-detects
        # its encoding from its weights during architecture resolution.

        # Validate the draft model config against its architecture
        # (quantization, rope type, encoding, etc.), then the target
        # against its own.
        self._validate_model_config_against_arch(self.draft_model, draft_arch)
        self._validate_model_config_against_arch(self.model, target_arch)

    def _populate_model_configs_from_archs(self) -> None:
        """Assigns each model's encoding, weight paths, and devices, then validates.

        A CPU-only encoding downcasts all-GPU ``device_specs`` to CPU,
        warning once per model. Also applies the arch-declared defaults and
        runs the arch-dependent validations.
        Must use the same architecture-selection inputs as the registry.
        A determinable architecture name with no registered architecture is
        an error; models whose architecture name cannot be determined keep
        their raw fields and are reported downstream.
        """
        if "main" not in self.models:
            return
        try:
            arch_name: str | None = self.models.main_architecture_name
        except Exception:
            logger.debug(
                "Could not determine the main architecture name at "
                "construction; skipping construction-time resolution.",
                exc_info=True,
            )
            arch_name = None
        task = (
            self.task
            if self.task != PipelineTask.UNDEFINED
            else PipelineTask.TEXT_GENERATION
        )
        arch = find_architecture(
            arch_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
            task=task,
        )
        if arch_name is not None and arch is None:
            # Custom architectures are imported before this lookup, so an
            # unregistered name is a hard error here. Only an undeterminable
            # name (a repo/metadata problem) defers to the downstream path.
            raise ValueError(f"No architecture found for {arch_name}")
        if arch is not None:
            _populate_weights_and_encoding(
                self.model,
                default_encoding=arch.default_encoding,
                supported_encodings=arch.supported_encodings,
                default_weights_format=arch.default_weights_format,
            )
        draft_arch = None
        if self.draft_model is not None:
            try:
                draft_arch_name: str | None = self.draft_model.architecture_name
            except Exception:
                logger.debug(
                    "Could not determine the draft architecture name at "
                    "construction; skipping construction-time resolution.",
                    exc_info=True,
                )
                draft_arch_name = None
            # Mirrors the registry's draft lookup, which passes no task.
            draft_arch = find_architecture(
                draft_arch_name,
                prefer_module_v3=self.runtime.prefer_module_v3,
            )
            if draft_arch_name is not None and draft_arch is None:
                raise ValueError(
                    "MAX-Optimized architecture not found for `draft_model`"
                )
            if draft_arch is not None:
                _populate_weights_and_encoding(
                    self.draft_model,
                    default_encoding=draft_arch.default_encoding,
                    supported_encodings=draft_arch.supported_encodings,
                    default_weights_format=draft_arch.default_weights_format,
                )

        if arch is None:
            return
        if not self.runtime.force:
            # Draft first so the target architecture wins conflicting keys,
            # matching the order resolve() historically applied them in.
            if draft_arch is not None:
                self._validate_required_arguments_against_architecture(
                    draft_arch
                )
            self._validate_required_arguments_against_architecture(arch)
        self._resolve_default_reasoning_parser(arch=arch)
        self._resolve_default_tool_parser(arch=arch)
        self._resolve_default_structured_output_backend(arch=arch)
        self._resolve_default_structured_output_any_whitespace(arch=arch)
        self._resolve_max_length(arch=arch, draft_arch=draft_arch)
        self._validate_synthetic_acceptance_with_constrained_decoding()

        if (
            self.sampling.enable_structured_output
            and self.model.default_device_spec.device_type == "cpu"
        ):
            raise ValueError(
                "enable_structured_output is not currently supported on CPU."
            )

        if self.draft_model is not None:
            # draft_arch is only None here when the draft's architecture
            # name could not be determined; the registry reports that.
            if draft_arch is not None:
                self._validate_speculative_model_configs(
                    target_arch=arch, draft_arch=draft_arch
                )
                self._validate_pipeline_config_for_speculative_decoding(
                    target_arch=arch,
                    draft_arch=draft_arch,
                )
        else:
            self._validate_model_config_against_arch(self.model, arch)

        self._validate_and_resolve_overlap_scheduler(arch=arch)

    # NOTE: Do not override `__getstate__` / `__setstate__` on Pydantic models.
    #
    # Pydantic's BaseModel implements a pickling protocol that expects a specific
    # state shape. Overriding `__getstate__` without also providing a compatible
    # `__setstate__` breaks unpickling (e.g. restores an "empty" model with
    # defaults).
    #
    # We still avoid pickling `transformers` objects via `MAXModelConfig`'s
    # custom pickling hooks (it drops `_huggingface_config`), so `PipelineConfig`
    # should rely on the BaseModel implementation.

    @classmethod
    def from_args(cls, args: PipelineArgs) -> Self:
        """Construct a :class:`PipelineConfig` from a :class:`PipelineArgs`.

        Args:
            args: Flat user-facing pipeline arguments.

        Returns:
            A fully constructed and validated :class:`PipelineConfig`.
        """
        # Register user-supplied custom architectures before any
        # construction-time architecture lookup (they may override built-ins).
        import_custom_architectures(args.runtime.custom_architectures)

        if args._manifest_override is not None:
            manifest = args._manifest_override
        else:
            models_dict: dict[str, MAXModelConfig] = {
                "main": MAXModelConfig.from_pipeline_args(args)
            }
            if args.draft_model is not None:
                models_dict["draft"] = args.draft_model.model_copy(deep=True)
            manifest = ModelManifest(models_dict)

        # The model's HF generation_config may declare default sampling
        # params (e.g. repetition_penalty) that the sampler can only honor
        # if the matching feature is compiled in. Build sampling from the
        # user-set fields only (Pydantic fields-set), then let
        # from_generation_config_sampling_defaults switch on
        # enable_penalties/enable_min_tokens where the generation config
        # requires them.
        if "main" in manifest:
            main_model = manifest["main"]
            explicit_sampling = args.sampling.model_dump(
                include=args.sampling.model_fields_set
                - {"config_file", "section_name"}
            )
            if main_model.enable_echo:
                explicit_sampling["enable_variable_logits"] = True
            sampling = SamplingConfig.from_generation_config_sampling_defaults(
                sampling_params_defaults=main_model.sampling_params_defaults,
                **explicit_sampling,
            )
        else:
            sampling = _construct_from_user_fields(args.sampling)

        # Apply --model-override entries to the manifest before construction
        # (with_override returns a new manifest). Idempotent for "main"/
        # "draft" fields that from_flat_kwargs already folded into the flat
        # fields; this is the only application path for pre-built manifests
        # and programmatically constructed PipelineArgs.
        for component, fields in _parse_component_overrides(
            args.model_override
        ).items():
            if component not in manifest:
                raise ValueError(
                    f"Component {component!r} not found in manifest. "
                    f"Available: {list(manifest.keys())}"
                )
            manifest = manifest.with_override(component, **fields)

        config = cls(
            models=manifest,
            model_override=list(args.model_override),
            sampling=sampling,
            runtime=_construct_from_user_fields(args.runtime),
            profiling=_construct_from_user_fields(args.profiling),
            lora=args.lora.model_copy(deep=True) if args.lora else None,
            speculative=args.speculative.model_copy(deep=True)
            if args.speculative
            else None,
            task=args.task,
            debug_verify_replay=args.debug_verify_replay,
        )

        config._apply_speculative_draft_architecture()
        # Must precede the arch lookups so every consumer resolves the
        # overridden arch (#88511). Best-effort: repos whose HF config
        # cannot load fail downstream instead.
        try:
            config._apply_speculative_target_architecture()
        except Exception:
            logger.debug(
                "Could not apply the speculative target-architecture "
                "override at construction.",
                exc_info=True,
            )
        config._validate_repo_access()
        config._populate_model_configs_from_archs()
        # Freeze the manifest: construction is complete, so any later dict
        # mutation must go through with_override() on a new manifest.
        config.models.resolve()
        # Overlap/DGC resolution above is arch-gated; configs without a
        # registered architecture still end with a concrete bool.
        if config.runtime.device_graph_capture is None:
            config.runtime.device_graph_capture = False
        return config


def _parse_flag_bool(value: str, flag_name: str) -> bool:
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError(
            f"Invalid boolean value: {value} for flag: {flag_name}"
        )


def _parse_flag_int(value: str, flag_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value: {value} for flag: {flag_name}"
        ) from exc


PrometheusMetricsMode = Literal[
    "instrument_only", "launch_server", "launch_multiproc_server"
]
"""Controls the Prometheus metrics mode.

``"instrument_only"``
    Instrument metrics through the Prometheus client library, relying on the
    application to handle the metrics server.
``"launch_server"``
    Launch a Prometheus server to handle metrics requests.
``"launch_multiproc_server"``
    Launch a Prometheus server in multiprocess mode to report metrics.
"""
