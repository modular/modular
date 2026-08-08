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
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.lib.interfaces import (
    ArchConfig,
    ArchConfigWithKVCache,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import (
    DISABLE_PARSER_SENTINEL,
    PipelineRuntimeConfig,
)
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.sampling import (
    DEFAULT_STRUCTURED_OUTPUT_BACKEND,
    SamplingConfig,
)
from max.pipelines.speculative.config import SpeculativeConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .model_config import (
    MAXModelConfig,
    _effective_device_specs,
    _parse_component_overrides,
    _select_dtype_cast,
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

    @model.setter
    def model(self, value: MAXModelConfig) -> None:
        self.models = self.models.with_override("main", config=value)

    @property
    def draft_model(self) -> MAXModelConfig | None:
        """The draft model configuration. Alias for ``models.get("draft")``."""
        return self.models.get("draft")

    @draft_model.setter
    def draft_model(self, value: MAXModelConfig) -> None:
        self.models = self.models.with_override("draft", config=value)

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
        per-GPU allocations used by P2P collectives. Each independent allocation
        site contributes one set of ``ngpus`` buffers. The base estimate counts
        the sites visible from :class:`PipelineConfig`:

        - main model graph (multi-GPU only),
        - :class:`BlockOffloadEngine` for KV-cache offloading, *only* when its
          ``replicate_kv_across_tp`` path is active (MLA model with DP=1 and
          multi-device TP). See ``block_copy_engine.py`` / ``transfer_engine.py``.

        Returns 0 for single-device pipelines.

        Args:
            arch_config: Optional architecture config. When provided and it
                exposes KV params, the BCE term is gated on the actual
                ``replicates_kv_across_tp`` flag rather than only the
                ``kv_connector`` setting. Without it, the BCE term is added
                whenever a connector is configured (conservative).

        Returns:
            Estimated total signal-buffer memory in bytes (across all devices).
        """
        ngpus = len(self.model.device_specs)
        if ngpus <= 1:
            return 0

        count_per_gpu = 1  # main model
        if self.model.kv_cache.kv_connector in {
            KVConnectorType.tiered,
            KVConnectorType.local,
            KVConnectorType.rust_tiered,
        }:
            # BlockOffloadEngine only allocates signal buffers when its
            # broadcast path is active (replicate_kv_across_tp = is_mla AND
            # dp==1 AND n_devices>1; see block_copy_engine.py:242-306).
            # Without arch_config we can't tell, so be conservative and add
            # the set; with arch_config, gate precisely.
            bce_allocates = True
            if isinstance(arch_config, ArchConfigWithKVCache):
                bce_allocates = (
                    arch_config.get_kv_params().replicates_kv_across_tp
                )
            if bce_allocates:
                count_per_gpu += 1  # BlockOffloadEngine

        return Signals.NUM_BYTES * count_per_gpu * ngpus

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

    def _resolve_speculative_target_architecture(self) -> None:
        """Override the target architecture for unified spec-decode pipelines.

        Unified EAGLE / DFlash / MTP pipelines fold the draft into a dedicated
        target architecture (e.g. ``DeepseekV3ForCausalLM`` →
        ``UnifiedMTPDeepseekV3ForCausalLM``). This mutates
        ``model.huggingface_config.architectures[0]`` in place.

        This must run *before* the architecture is resolved from
        ``models.main_architecture_name`` (i.e. before :meth:`resolve` is
        called), so that the resolved ``arch`` — consumed by memory estimation,
        the overlap scheduler, parser resolution, and ``pipeline_model``
        construction — reflects the override. The registry invokes it at that
        point. It is a no-op when speculative decoding is disabled.
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
        # Gemma 4 12B ships as the "gemma4_unified" model line; its DSpark
        # block drafter declares architectures: ["Gemma4DSparkModel"].
        if target_archs[0] == "Gemma4UnifiedForConditionalGeneration":
            draft_archs = (
                self.draft_model.huggingface_config.architectures
                if self.draft_model is not None
                else None
            )
            if draft_archs and draft_archs[0] == "Gemma4DSparkModel":
                target_archs[0] = "UnifiedDSparkGemma4ForCausalLM"
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

    def resolve(
        self,
        arch: Any,
        draft_arch: Any = None,
    ) -> None:
        """Validates the config.

        Args:
            arch: Pre-resolved target architecture from the registry.
            draft_arch: Pre-resolved draft architecture (speculative decoding
                only). Required when ``draft_model`` is set.
        """
        self.models.resolve()
        # Diffusers pipelines don't have a "main" model — they have
        # per-component configs (unet, vae, etc.).  The LLM-specific
        # validations below all assume a single main model, so skip
        # them for multi-component diffusers manifests.
        if "main" not in self.models:
            return

        # Validation for max_length is handled in MAXModelConfig

        if (
            self.sampling.enable_structured_output
            and self.model.default_device_spec.device_type == "cpu"
        ):
            raise ValueError(
                "enable_structured_output is not currently supported on CPU."
            )

        if self.sampling.enable_penalties and self.draft_model:
            logger.warning(
                "frequency_penalty, presence_penalty and repetition_penalty are not currently supported with speculative decoding."
            )
            self.sampling.enable_penalties = False

        # Validate LoRA compatibility with model configuration
        if self.lora and self.lora.enable_lora:
            self.model.validate_lora_compatibility()

        # NOTE: the unified spec-decode target-architecture override
        # (``_resolve_speculative_target_architecture``) is applied by the
        # registry *before* it resolves ``arch`` and passes it in here, so that
        # the ``arch`` consumed by memory estimation, the overlap scheduler, and
        # parser resolution below already reflects the override. Applying it
        # here (after ``arch`` is resolved) would leave those consumers using
        # the stale pre-override architecture. See SERVOPT regression from
        # PipelineConfig/registry decoupling (#88511).

        # By this point, we should have a valid model_path.

        if self.draft_model:
            self._validate_speculative_model_configs(
                target_arch=arch, draft_arch=draft_arch
            )
            self._validate_pipeline_config_for_speculative_decoding(
                target_arch=arch,
                draft_arch=draft_arch,
            )
        else:
            self._validate_remaining_pipeline_config(
                model_config=self.model, resolved_arch=arch
            )

        self._resolve_default_reasoning_parser(arch=arch)
        self._resolve_default_tool_parser(arch=arch)
        self._resolve_default_structured_output_backend(arch=arch)

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

        Runs unconditionally so the field is always a concrete ``str`` after
        ``resolve()``. The ``None`` sentinel (unset) is what distinguishes an
        explicit user value from the default -- mirroring the reasoning/tool
        parser resolvers above.
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

    def _validate_and_resolve_overlap_scheduler(
        self, arch: Any = None, max_batch_size: int = 1
    ) -> None:
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
                    "Automatically enabling device graph capture for %s with max_batch_size=%d. "
                    "You can manually disable this by setting --no-device-graph-capture.",
                    arch.name,
                    max_batch_size,
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
            if self._effective_device_type(arch) == "cpu":
                raise ValueError(
                    "Overlap scheduler is not supported with CPU models."
                )

    def _effective_device_type(self, arch: Any) -> str:
        """Returns the device type the main model actually runs on.

        Uses the resolved device specs (the raw ``device_specs`` field may
        differ when a CPU-only encoding downcasts defaulted GPU devices).
        Falls back to the raw field when no arch is available to resolve the
        encoding against.
        """
        if arch is None:
            return self.model.device_specs[0].device_type
        return _effective_device_specs(self.model, arch.default_encoding)[
            0
        ].device_type

    def _is_eligible_for_overlap_serve_optimizations(self, arch: Any) -> bool:
        # Overlap scheduling and device graph capture are only supported for
        # text generation. Auto-enabling them for other tasks (e.g. embeddings)
        # would fail downstream pipeline construction. See
        # `get_pipeline_for_task` in registry.py.
        return (
            arch.task == PipelineTask.TEXT_GENERATION
            and not self.sampling.enable_variable_logits
            and not self.lora
            and self._effective_device_type(arch) != "cpu"
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
        """Validates and resolves model config fields against a resolved architecture.

        Validates quantization encoding, rope type, LoRA support, multi-GPU
        compatibility, and encoding support. Mutates ``model_config`` in place
        (resolves encoding, cache dtype, rope type, weight path). Does not
        perform memory estimation.

        Args:
            model_config: The model configuration to validate and mutate.
            arch: The pre-resolved architecture to validate against.
        """
        # Validate required arguments
        if not self.runtime.force:
            self._validate_required_arguments_against_architecture(arch)

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
            if (
                len(
                    _effective_device_specs(model_config, arch.default_encoding)
                )
                > 1
            ):
                raise ValueError(
                    "LoRA is currently not supported with the number of devices > 1."
                )

        model_config.validate_multi_gpu_supported(
            multi_gpu_supported=arch.multi_gpu_supported
        )

        resolved_encoding = _select_quantization_encoding(
            model_config, arch.default_encoding
        )
        cast_from, _ = _select_dtype_cast(model_config, arch.default_encoding)
        if resolved_encoding not in arch.supported_encodings:
            raise ValueError(
                f"quantization_encoding of '{resolved_encoding}' not supported by MAX engine."
            )
        model_config.validate_and_resolve_with_resolved_quantization_encoding(
            resolved_encoding=resolved_encoding,
            applied_dtype_cast_from=cast_from,
            default_weights_format=arch.default_weights_format,
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

        # Validate draft model config against its architecture (quantization,
        # rope type, encoding, etc.). Target validation is handled inside
        # _validate_remaining_pipeline_config below.
        self._validate_model_config_against_arch(self.draft_model, draft_arch)
        self._validate_remaining_pipeline_config(
            model_config=self.model,
            resolved_arch=target_arch,
        )

    def _validate_remaining_pipeline_config(
        self,
        model_config: MAXModelConfig,
        resolved_arch: Any,
    ) -> None:
        """Validates model config against the architecture.

        Memory estimation and max_length resolution have moved to the registry
        (``retrieve_factory``), where they run after this validation completes.

        Args:
            model_config: The model configuration to validate.
            resolved_arch: Pre-resolved architecture from the registry.
        """
        self._validate_model_config_against_arch(model_config, resolved_arch)

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
            A fully constructed :class:`PipelineConfig` ready for
            architecture-driven resolution via :meth:`resolve`.
        """
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
        config._validate_repo_access()
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
