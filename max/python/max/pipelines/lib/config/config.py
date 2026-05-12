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

import importlib
import json
import logging
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Literal, cast

from max.config import ConfigFileModel
from max.driver import DeviceSpec, accelerator_api, load_devices
from max.engine import InferenceSession
from max.graph.quantization import QuantizationEncoding
from max.interfaces.task import PipelineTask
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.lib.hf_utils import is_diffusion_pipeline
from max.pipelines.lib.interfaces import PipelineModel
from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig
from max.pipelines.lib.memory_estimation import (
    MemoryEstimator,
    to_human_readable_bytes,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.registry import (
    PIPELINE_REGISTRY,
    SupportedArchitecture,
    get_pipeline_for_task,
)
from max.pipelines.lib.sampling import SamplingConfig
from pydantic import (
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from typing_extensions import Self, override

from .kv_cache_config import KVCacheConfig, KVConnectorConfig
from .lora_config import LoRAConfig
from .model_config import MAXModelConfig, _format_config_entries
from .profiling_config import ProfilingConfig
from .speculative_config import SpeculativeConfig

logger = logging.getLogger("max.pipelines")

# ModelManifest is a dict[str, MAXModelConfig] subclass with extra methods.
# cyclopts (CLI framework) only recognizes plain dict types via typing.get_origin(),
# which returns None for concrete subclasses. At runtime, Pydantic sees
# dict[str, MAXModelConfig] so cyclopts can resolve CLI paths like
# --pipeline.models.main.model-path. mypy sees ModelManifest so methods like
# .with_override(), .resolve(), .main_architecture_name type-check correctly.
if TYPE_CHECKING:
    _ModelsType = ModelManifest
else:
    _ModelsType = dict[str, MAXModelConfig]


def _strip_default_model_kwargs(
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return *model_kwargs* with entries that match MAXModelConfig defaults removed.

    Fields declared with ``default_factory`` have ``field.default`` set to
    ``PydanticUndefined``, so we must invoke the factory to obtain the
    comparable default value.
    """
    from pydantic_core import PydanticUndefined

    fields = MAXModelConfig.model_fields
    non_default: dict[str, Any] = {}
    for k, v in model_kwargs.items():
        field = fields.get(k)
        if field is None:
            # Not a MAXModelConfig field — keep it.
            non_default[k] = v
            continue
        if field.default is not PydanticUndefined:
            if v == field.default:
                continue
        elif field.default_factory is not None:
            try:
                if v == field.default_factory():  # type: ignore[call-arg]
                    continue
            except Exception:
                pass
        non_default[k] = v
    return non_default


# FIXME: This method seems like a major hack...
# Can this be moved to the KVCacheConfig post init?
def _resolve_kvconnector_config(kv: KVCacheConfig) -> None:
    """Validates KV connector configuration and applies defaults."""
    connector = kv.kv_connector
    if connector is None:
        return

    # Ensure a config object exists for connectors that need one.
    cfg = kv.kv_connector_config or KVConnectorConfig()

    if connector == KVConnectorType.tiered:
        if cfg.disk_offload_dir is None:
            cfg.disk_offload_dir = tempfile.mkdtemp(prefix="max_kv_tiered_")
            logger.info(
                f"Tiered connector: auto-created disk offload dir "
                f"{cfg.disk_offload_dir}"
            )

    kv.kv_connector_config = cfg


_AUTO_ENABLE_OVERLAP_SCHEDULER_ARCHITECTURES = (
    "LlamaForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLMNextN",
    "KimiK25ForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "UnifiedEagleLlama3ForCausalLM",
    "UnifiedMTPDeepseekV3ForCausalLM",
    "Eagle3DeepseekV2ForCausalLM",
    "Eagle3DeepseekV3ForCausalLM",
    "MiniMaxM2ForCausalLM",
)

_AUTO_ENABLE_DEVICE_GRAPH_CAPTURE_ARCHITECTURES = (
    "LlamaForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLMNextN",
    "KimiK25ForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "UnifiedEagleLlama3ForCausalLM",
    "UnifiedMTPDeepseekV3ForCausalLM",
    "Eagle3DeepseekV2ForCausalLM",
    "Eagle3DeepseekV3ForCausalLM",
    "MiniMaxM2ForCausalLM",
)


class PipelineConfig(ConfigFileModel):
    """Configuration for a pipeline.

    Contains settings for model selection, batch sizing, sampling, profiling,
    LoRA adapters, and speculative decoding. Once initialized, all fields are
    resolved to their final values from CLI flags, config files, environment
    variables, or internal defaults.
    """

    # PipelineConfig intentionally accepts kwargs that belong to sub-configs
    # (MAXModelConfig, KVCacheConfig, etc.) and routes them via the
    # _preprocess_kwargs wrap validator.  Allow extras so pydantic (and its
    # mypy plugin) don't reject those unmatched kwargs.
    # TODO: This should be removed though, but only after we've fully unrolled
    # the weird monkeypatching to instantiate MAXModelConfig, KVCacheConfig, etc.
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

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
        """
        result: dict[str, Any] = {}
        for role, value in data.items():
            if isinstance(value, dict):
                result[role] = {
                    k.replace("-", "_"): v for k, v in value.items()
                }
            else:
                result[role] = value
        return result

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

    _config_file_section_name: str = PrivateAttr(default="pipeline_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    _unmatched_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Temporary storage for unmatched kwargs during initialization.
    This is used to pass unmatched kwargs from the before validator to the after validator."""

    def configure_session(self, session: InferenceSession) -> None:
        """Configures a :class:`~max.engine.InferenceSession` with standard pipeline settings."""
        session.gpu_profiling(self.profiling.gpu_profiling)
        session._use_experimental_kernels(self.runtime.use_experimental_kernels)
        session._use_vendor_blas(self.runtime.use_vendor_blas)
        session._use_vendor_ccl(self.runtime.use_vendor_ccl)

    @staticmethod
    def _extract_kwargs_for_config(
        kwargs: dict[str, Any],
        config_class: type[ConfigFileModel],
        key_prefix: str = "",
        strip_prefix: bool = False,
    ) -> dict[str, Any]:
        """Extracts kwargs that match a config class's fields.

        Args:
            kwargs: Source kwargs dictionary (modified in place)
            config_class: The ConfigFileModel dataclass to match fields against
            key_prefix: Optional prefix to filter keys (for example, ``"draft_"``)
            strip_prefix: Whether to strip the prefix from extracted keys

        Returns:
            Dictionary of extracted kwargs
        """
        extracted = {}
        keys_to_remove = []

        for key, value in kwargs.items():
            # Check if key matches the prefix filter
            if key_prefix and not key.startswith(key_prefix):
                continue

            # Determine the field name to check
            field_name = key.replace(key_prefix, "") if strip_prefix else key

            # Check if this field exists in the config class (Pydantic model)
            if field_name in config_class.model_fields:
                # Use original key or stripped key as specified
                extracted_key = field_name if strip_prefix else key
                extracted[extracted_key] = value
                keys_to_remove.append(key)

        # Remove extracted keys from original kwargs
        for key in keys_to_remove:
            del kwargs[key]

        return extracted

    def _create_denoising_cache_config_if_needed(
        self, kwargs: dict[str, Any]
    ) -> None:
        """Extract denoising cache kwargs and set on runtime.denoising_cache."""
        cache_kwargs = PipelineConfig._extract_kwargs_for_config(
            kwargs, DenoisingCacheConfig
        )
        if cache_kwargs:
            # Remove None values so DenoisingCacheConfig defaults are used
            filtered = {k: v for k, v in cache_kwargs.items() if v is not None}
            if filtered:
                self.runtime.denoising_cache = DenoisingCacheConfig(**filtered)

    def _create_lora_config_if_needed(self, kwargs: dict[str, Any]) -> None:
        """Extract LoRA kwargs and create valid LoRAConfig if enable_lora provided."""
        lora_kwargs = PipelineConfig._extract_kwargs_for_config(
            kwargs, LoRAConfig
        )

        if lora_kwargs.get("enable_lora", False):
            self.lora = LoRAConfig(**lora_kwargs)
        # TODO: We should add an elif to check / error out if other LoRA params
        # are provided, but enable_lora is not. We can't do this today as our
        # click PipelineConfig autogenerates defaults for all fields, including
        # required ones.

    def _build_models_from_kwargs(
        self, unmatched_kwargs: dict[str, Any]
    ) -> None:
        """Build the ModelManifest from unmatched model kwargs.

        Uses ``ModelManifest.from_model_path()`` as the single entry point
        for creating the main model config. Handles KV cache kwargs
        separately and adds draft model via ``with_override``.

        When the manifest is already populated (e.g. passed directly via
        ``models=``), this method only processes KV cache and draft kwargs
        without rebuilding the manifest.
        """
        # Extract model kwargs (model_path, quantization_encoding, etc.)
        model_kwargs = PipelineConfig._extract_kwargs_for_config(
            unmatched_kwargs, MAXModelConfig
        )
        kv_cache_kwargs: dict[str, Any] = {}
        for key in list(unmatched_kwargs):
            if key in KVCacheConfig.model_fields:
                kv_cache_kwargs[key] = unmatched_kwargs.pop(key)

        # Parse --model-override entries once, grouped by target component, so
        # "main"/"draft" overrides can be folded into the constructor kwargs
        # below.  HuggingFaceRepo.__post_init__ and the MAXModelConfig
        # validator eagerly hit HF using the dataclass-default revision, which
        # fails under HF_HUB_OFFLINE for repos cached only at a pinned SHA.
        component_overrides: dict[str, dict[str, Any]] = {}
        for override_str in self.model_override:
            component, field_name, value = self._parse_model_override(
                override_str
            )
            component_overrides.setdefault(component, {})[field_name] = value

        # Only rebuild the manifest when explicit model kwargs were provided
        # in unmatched_kwargs.  When a pre-built ModelManifest was passed via
        # the ``models=`` kwarg, ``model_kwargs`` will be empty and we should
        # not reconstruct the manifest (which would trigger HF validation).
        if model_kwargs:
            model_kwargs.update(component_overrides.get("main", {}))
            model_path = model_kwargs.pop("model_path", "")
            if model_path:
                revision = model_kwargs.pop("huggingface_model_revision", None)
                # Strip kwargs that match MAXModelConfig defaults so
                # from_model_path() doesn't reject them for diffusion
                # pipelines (which forbid extra kwargs).
                non_default_kwargs = _strip_default_model_kwargs(model_kwargs)
                self.models = ModelManifest.from_model_path(
                    model_path,
                    revision=revision,
                    **non_default_kwargs,
                )

        # Apply KV cache config to main model
        if kv_cache_kwargs and "main" in self.models:
            self.model.create_kv_cache_config(**kv_cache_kwargs)

        # Extract draft model kwargs and add via with_override
        draft_kwargs = PipelineConfig._extract_kwargs_for_config(
            unmatched_kwargs,
            MAXModelConfig,
            key_prefix="draft_",
            strip_prefix=True,
        )
        if draft_kwargs.get("model_path", "") != "":
            # Inherit certain fields from the target model if not explicitly
            # specified for the draft model. This simplifies CLI usage for
            # speculative decoding (e.g. --draft-trust-remote-code is not
            # needed if --trust-remote-code is already set).
            if "main" in self.models:
                self._apply_draft_model_defaults(draft_kwargs, self.model)

            # "draft" overrides are applied after inheritance so explicit
            # user intent wins over copied target-model defaults.
            draft_kwargs.update(component_overrides.get("draft", {}))

            draft_config = MAXModelConfig(**draft_kwargs)
            if kv_cache_kwargs:
                draft_config.create_kv_cache_config(**kv_cache_kwargs)
            self.models = self.models.with_override(
                "draft", config=draft_config
            )

        # Apply parsed overrides via with_override.  This is idempotent for
        # "main"/"draft" fields already folded into kwargs above, and is the
        # only path that runs when a pre-built manifest was passed via
        # ``models=``.
        for component, fields in component_overrides.items():
            if component not in self.models:
                raise ValueError(
                    f"Component {component!r} not found in manifest. "
                    f"Available: {list(self.models.keys())}"
                )
            self.models = self.models.with_override(component, **fields)

    @staticmethod
    def _apply_draft_model_defaults(
        draft_kwargs: dict[str, Any], target_model: MAXModelConfig
    ) -> None:
        """Inherit certain fields from the target model for the draft model.

        When running speculative decoding, the draft model typically shares
        configuration with the target model (same devices, same trust settings,
        same parallelism). This method copies these fields from the target
        model config into the draft kwargs if they weren't explicitly specified.

        Fields inherited:
        - ``trust_remote_code``: If the target model requires custom code,
          the draft model (from the same model family) likely does too.
        - ``device_specs``: The draft model runs on the same devices as the
          target model.
        - ``data_parallel_degree``: Both models use the same parallelism.

        Note: ``quantization_encoding`` is NOT inherited because draft models
        (especially EAGLE3) often use bfloat16 regardless of the target model's
        quantization. The draft model should auto-detect its encoding from its
        weights.

        Args:
            draft_kwargs: The draft model kwargs dict (modified in place).
            target_model: The target model configuration to inherit from.
        """
        # Inherit trust_remote_code if not explicitly specified
        if "trust_remote_code" not in draft_kwargs:
            if target_model.trust_remote_code:
                logger.info(
                    "Inheriting trust_remote_code=True from target model "
                    "for draft model"
                )
                draft_kwargs["trust_remote_code"] = True

        # Inherit device_specs if not explicitly specified
        if "device_specs" not in draft_kwargs:
            logger.info(
                f"Inheriting device_specs={target_model.device_specs} "
                "from target model for draft model"
            )
            draft_kwargs["device_specs"] = target_model.device_specs

        # Inherit data_parallel_degree if not explicitly specified
        if "data_parallel_degree" not in draft_kwargs:
            if target_model.data_parallel_degree != 1:
                logger.info(
                    f"Inheriting data_parallel_degree="
                    f"{target_model.data_parallel_degree} from target model "
                    "for draft model"
                )
            draft_kwargs["data_parallel_degree"] = (
                target_model.data_parallel_degree
            )

    @staticmethod
    def _parse_model_override(override_str: str) -> tuple[str, str, Any]:
        """Parse ``component.field=value`` into ``(component, field, value)``.

        The value is coerced to the target field's type via Pydantic's
        ``TypeAdapter`` (JSON-first, raw-string fallback for scalars).

        Raises:
            ValueError: if the string is malformed or names an unknown
                ``MAXModelConfig`` field.
        """
        dot_pos = override_str.find(".")
        if dot_pos < 1:
            raise ValueError(
                f"Invalid --model-override format: {override_str!r}. "
                f"Expected 'component.field=value'."
            )
        eq_pos = override_str.find("=", dot_pos)
        if eq_pos < dot_pos + 2:
            raise ValueError(
                f"Invalid --model-override format: {override_str!r}. "
                f"Expected 'component.field=value'."
            )
        component = override_str[:dot_pos]
        field_name = override_str[dot_pos + 1 : eq_pos]
        raw_value = override_str[eq_pos + 1 :]

        if field_name not in MAXModelConfig.model_fields:
            raise ValueError(
                f"Unknown MAXModelConfig field: {field_name!r}. "
                f"Valid fields: {sorted(MAXModelConfig.model_fields.keys())}"
            )

        # For compound types (list, dict) the raw CLI string is JSON, so try
        # json.loads first; fall back to the raw string for plain scalars.
        field_info = MAXModelConfig.model_fields[field_name]
        adapter: TypeAdapter[Any] = TypeAdapter(field_info.annotation)
        try:
            parsed_value = json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            parsed_value = raw_value
        return component, field_name, adapter.validate_python(parsed_value)

    def _create_speculative_config_if_needed(
        self, kwargs: dict[str, Any]
    ) -> None:
        """Extract speculative config kwargs and create SpeculativeConfig if any speculative parameters provided."""
        speculative_kwargs = PipelineConfig._extract_kwargs_for_config(
            kwargs, SpeculativeConfig
        )
        # Only create speculative config if speculative_method is explicitly set
        if not (
            speculative_kwargs
            and speculative_kwargs.get("speculative_method") is not None
        ):
            return

        # Remove None values to use defaults
        filtered_kwargs = {
            k: v for k, v in speculative_kwargs.items() if v is not None
        }
        if not filtered_kwargs:
            return

        self.speculative = SpeculativeConfig(**filtered_kwargs)
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

    # Explicit type mapping for config classes that are processed from
    # unmatched kwargs.  "model" is handled separately in
    # _build_models_from_kwargs via ModelManifest.from_model_path().
    _CONFIG_TYPE_MAPPING: dict[str, type[ConfigFileModel]] = {
        "runtime": PipelineRuntimeConfig,
        "sampling": SamplingConfig,
        "profiling": ProfilingConfig,
    }

    def _process_remaining_config_classes(
        self, unmatched_kwargs: dict[str, Any]
    ) -> None:
        """Processes remaining kwargs for other config classes.

        Note: model kwargs are handled separately in ``_build_models_from_kwargs``.

        Args:
            unmatched_kwargs: Dictionary of kwargs that haven't been matched yet
        """
        # NOTE: runtime must come before sampling so that its
        # fields are consumed first.
        # NOTE: model must be built before sampling so that
        # SamplingConfig can use generation_config from the model.
        # Model is handled in _build_models_from_kwargs, which runs
        # before this method.
        config_mappings = ["runtime", "sampling", "profiling"]

        for config_name in config_mappings:
            config_class = self._CONFIG_TYPE_MAPPING[config_name]
            matched_kwargs = {}

            for key, value in unmatched_kwargs.items():
                if key in config_class.model_fields:
                    matched_kwargs[key] = value

            if matched_kwargs:
                self._create_and_set_config(
                    config_name, config_class, matched_kwargs
                )

                # Remove matched kwargs
                for key in matched_kwargs:
                    _ = unmatched_kwargs.pop(key, None)

    def _create_and_set_config(
        self,
        config_name: str,
        config_class: type,
        matched_kwargs: dict[str, Any],
    ) -> None:
        """Creates and sets a config object with special handling for config types.

        Args:
            config_name: Name of the config attribute (for example, ``"sampling"``)
            config_class: The config class to instantiate
            matched_kwargs: kwargs that matched the config class fields
        """
        if config_name == "sampling":
            if "main" in self.models:
                assert isinstance(self.model, MAXModelConfig)
                assert hasattr(
                    config_class, "from_generation_config_sampling_defaults"
                )
                sampling_config = config_class.from_generation_config_sampling_defaults(
                    sampling_params_defaults=self.model.sampling_params_defaults,
                    **matched_kwargs,
                )
            else:
                sampling_config = config_class(**matched_kwargs)

            is_standalone_spec_decoding = (
                self.speculative and self.speculative.is_standalone()
            )
            if (
                "main" in self.models and self.model.enable_echo
            ) or is_standalone_spec_decoding:
                sampling_config.enable_variable_logits = True
            setattr(self, config_name, sampling_config)
        else:
            setattr(self, config_name, config_class(**matched_kwargs))

    # This has to be mode="wrap" instead of mode="before" to be able to pass
    # state of self._unmatched_kwargs to be used in the mode="after" validator
    # function given it's a PrivateAttr.
    @model_validator(mode="wrap")
    @classmethod
    def _preprocess_kwargs(
        cls, data: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Preprocess kwargs before Pydantic validation.

        We need to separate kwargs for nested configs *and* pass the unmatched
        kwargs through to the post-processing validator. Since `_unmatched_kwargs`
        is a `PrivateAttr`, it cannot be set via normal model input, so we use a
        wrap validator and stash the unmatched values onto the instance after
        Pydantic has created it.
        """
        if not isinstance(data, dict):
            return handler(data)

        kwargs = data.copy()
        # Merge config file values before separating pydantic vs unmatched
        # kwargs, so sub-config fields (e.g. model_path) from the YAML are
        # visible to _postprocess_configs.
        kwargs = cls.load_config_file(kwargs)  # type: ignore[operator]

        # Intercept model/draft_model before field separation — these are
        # no longer Pydantic fields but consumers still pass them directly.
        model_kwarg = kwargs.pop("model", None)
        draft_model_kwarg = kwargs.pop("draft_model", None)

        # If a MAXModelConfig (or plain dict from config file) was passed
        # directly, wrap it in a manifest.  Coerce dicts so that callers
        # loading from YAML/JSON (which produce plain dicts) work correctly.
        if model_kwarg is not None:
            if isinstance(model_kwarg, dict) and not isinstance(
                model_kwarg, MAXModelConfig
            ):
                model_kwarg = MAXModelConfig(**model_kwarg)
            kwargs["models"] = ModelManifest({"main": model_kwarg})

        unmatched_kwargs: dict[str, Any] = {}
        # Use getattr to safely access model_fields in case it's not yet available
        # during class construction.
        model_fields = getattr(cls, "model_fields", {})

        # Separate kwargs that belong to this class vs other config classes.
        pydantic_kwargs: dict[str, Any] = {}
        for key, value in list(kwargs.items()):
            if key in model_fields:
                pydantic_kwargs[key] = value
                logger.debug("pydantic_kwargs key: %s, value: %s", key, value)
            else:
                unmatched_kwargs[key] = value
                logger.debug("unmatched_kwargs key: %s, value: %s", key, value)

        instance = handler(pydantic_kwargs)
        # `_unmatched_kwargs` is a PrivateAttr, so set it on the instance.
        instance._unmatched_kwargs = unmatched_kwargs

        # Add draft model via with_override
        if draft_model_kwarg is not None:
            if isinstance(draft_model_kwarg, dict) and not isinstance(
                draft_model_kwarg, MAXModelConfig
            ):
                draft_model_kwarg = MAXModelConfig(**draft_model_kwarg)
            instance.models = instance.models.with_override(
                "draft", config=draft_model_kwarg
            )

        return instance

    @model_validator(mode="after")
    def __postprocess_configs(self) -> Self:
        """Process nested configs after Pydantic validation.

        This runs after all fields have been validated and set.
        """
        # Get unmatched kwargs that were stored during preprocessing
        try:
            unmatched_kwargs = self._unmatched_kwargs
        except AttributeError:
            # Pydantic re-validates 'after' validators when placed inside
            # another model, at which point _postprocess_configs will have
            # already run once and _unmatched_kwargs won't be set.  We don't
            # need to run again in this case.
            return self
        delattr(self, "_unmatched_kwargs")

        # Process specialized config creation
        self._create_lora_config_if_needed(unmatched_kwargs)

        # Build model manifest from kwargs — must come before sampling
        # (which needs model's generation_config) and speculative
        # (which needs draft_model).
        self._build_models_from_kwargs(unmatched_kwargs)
        self._create_speculative_config_if_needed(unmatched_kwargs)

        # Process remaining config classes (runtime, sampling, profiling)
        if unmatched_kwargs:
            self._process_remaining_config_classes(unmatched_kwargs)

        # Set denoising_cache on runtime AFTER runtime is constructed by
        # _process_remaining_config_classes; otherwise the runtime
        # replacement there clobbers the cache fields set here.
        self._create_denoising_cache_config_if_needed(unmatched_kwargs)

        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        # Check both the defer_resolve field and the environment variable
        defer_resolve_env = os.getenv(
            "MODULAR_PIPELINE_DEFER_RESOLVE", ""
        ).lower()
        should_defer = self.runtime.defer_resolve or defer_resolve_env in {
            "1",
            "true",
            "yes",
        }
        if not should_defer:
            self.resolve()
        return self

    def _import_custom_architectures(self) -> None:
        """Imports custom model modules and adds them to the registry."""
        for module_spec in self.runtime.custom_architectures:
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
                PIPELINE_REGISTRY.register(arch, allow_override=True)

    def _validate_required_arguments_against_architecture(
        self, architecture: SupportedArchitecture
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

    def resolve(self) -> None:
        """Validates and resolves the config.

        Called after the config is initialized to ensure all config fields
        are in a valid state.
        """
        # Before anything else, import custom model modules to add them to the registry.
        self._import_custom_architectures()

        self.models.resolve()
        # Diffusers pipelines don't have a "main" model — they have
        # per-component configs (unet, vae, etc.).  The LLM-specific
        # validations below all assume a single main model, so skip
        # them for multi-component diffusers manifests.
        if "main" not in self.models:
            return

        # Validation for max_length is handled in MAXModelConfig

        self._validate_and_resolve_max_num_steps()

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

        # Override target architecture for unified EAGLE pipeline.
        if self.speculative:
            target_archs = self.model.huggingface_config.architectures
            if target_archs[0] == "LlamaForCausalLM":
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
                    draft_archs
                    and draft_archs[0] == "Eagle3DeepseekV2ForCausalLM"
                ):
                    target_archs[0] = "Eagle3DeepseekV3ForCausalLM"
                else:
                    if not draft_archs:
                        raise ValueError(
                            "Draft model HF config has empty"
                            " ``architectures=[]``. Expected"
                            " 'Eagle3DeepseekV2ForCausalLM' (Eagle3 draft) or"
                            " no draft model (MTP path)."
                        )
                    raise ValueError(
                        "Unrecognized draft architecture for DeepseekV3"
                        f" target: {draft_archs[0]!r}. Expected"
                        " 'Eagle3DeepseekV2ForCausalLM' (Eagle3 draft) or no"
                        " draft model (MTP path)."
                    )
            if target_archs[0] == "KimiK25ForConditionalGeneration":
                target_archs[0] = "Eagle3DeepseekV2ForCausalLM"

        # Validate KV connector configuration
        _resolve_kvconnector_config(self.model.kv_cache)

        # By this point, we should have a valid model_path.

        if self.draft_model:
            # Joint memory estimation for speculative decoding
            _resolve_kvconnector_config(self.draft_model.kv_cache)
            self._validate_and_resolve_speculative_memory()
            self._validate_pipeline_config_for_speculative_decoding()
        else:
            self._validate_and_resolve_remaining_pipeline_config(
                model_config=self.model
            )

        self._validate_and_resolve_overlap_scheduler()

        self._resolve_default_reasoning_parser()
        self._resolve_default_tool_parser()

    def _resolve_default_reasoning_parser(self) -> None:
        """Apply the architecture's default reasoning parser when unset.

        If the user did not configure ``runtime.reasoning_parser`` and the
        resolved ``SupportedArchitecture`` declares a default
        ``reasoning_parser``, use it. Explicit user configuration always wins.
        """
        if self.runtime.reasoning_parser is not None:
            return

        arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=self.models.main_architecture_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
        )
        if arch is None or arch.reasoning_parser is None:
            return

        self.runtime.reasoning_parser = arch.reasoning_parser
        logger.info(
            "Defaulting reasoning parser to %r for architecture %s. "
            "Override with --reasoning-parser.",
            arch.reasoning_parser,
            arch.name,
        )

    def _resolve_default_tool_parser(self) -> None:
        """Apply the architecture's default tool parser when unset.

        If the user did not configure ``runtime.tool_parser`` and the
        resolved ``SupportedArchitecture`` declares a default
        ``tool_parser``, use it. Explicit user configuration always wins.
        """
        if self.runtime.tool_parser is not None:
            return

        arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=self.models.main_architecture_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
        )
        if arch is None or arch.tool_parser is None:
            return

        self.runtime.tool_parser = arch.tool_parser
        logger.info(
            "Defaulting tool parser to %r for architecture %s. "
            "Override with --tool-parser.",
            arch.tool_parser,
            arch.name,
        )

    def _validate_and_resolve_overlap_scheduler(self) -> None:
        arch: SupportedArchitecture | None = None
        if not self.runtime.force:
            arch = PIPELINE_REGISTRY.retrieve_architecture(
                architecture_name=self.models.main_architecture_name,
                prefer_module_v3=self.runtime.prefer_module_v3,
            )
            max_batch_size = self.runtime.max_batch_size
            if (
                self.runtime.device_graph_capture is None
                and arch is not None
                and arch.name in _AUTO_ENABLE_DEVICE_GRAPH_CAPTURE_ARCHITECTURES
                and max_batch_size is not None
                and accelerator_api() == "cuda"
                and self._is_eligible_for_overlap_serve_optimizations()
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

        # Automatically enable overlap scheduling for select architectures.
        if not self.runtime.enable_overlap_scheduler:
            if (
                arch is not None
                and arch.name in _AUTO_ENABLE_OVERLAP_SCHEDULER_ARCHITECTURES
                and self._is_eligible_for_overlap_serve_optimizations()
            ):
                self.runtime.enable_overlap_scheduler = True
                self.runtime.max_num_steps = 1
                logger.info(
                    f"Automatically enabling overlap scheduling for {arch.name} with max-num-steps=1. "
                    "You can manually disable this by setting --no-enable-overlap-scheduler --force."
                )

        # Raise errors when we detect features that are not compatible with the overlap scheduler.
        if self.runtime.enable_overlap_scheduler:
            if self.runtime.pipeline_role in ("decode_only", "prefill_only"):
                if self.runtime.max_num_steps != 1:
                    logger.info(
                        "Setting max-num-steps=1 for overlap scheduling on %s worker.",
                        self.runtime.pipeline_role,
                    )
                    self.runtime.max_num_steps = 1
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
            if self.runtime.max_num_steps > 1:
                raise ValueError(
                    "Max num steps > 1 is not supported with the Overlap scheduler."
                )
            if self.model.device_specs[0].device_type == "cpu":
                raise ValueError(
                    "Overlap scheduler is not supported with CPU models."
                )

    def _is_eligible_for_overlap_serve_optimizations(self) -> bool:
        return (
            not self.sampling.enable_variable_logits
            and not self.lora
            and self.model.device_specs[0].device_type != "cpu"
        )

    def _validate_and_resolve_device_graph_capture(self) -> None:
        if not self.runtime.device_graph_capture:
            return

        if self.runtime.max_batch_size is None:
            raise ValueError(
                "device_graph_capture requires max_batch_size to be set."
            )
        if not self.runtime.enable_overlap_scheduler:
            logger.info("Enabling overlap scheduling for device graph capture.")
        self.runtime.enable_overlap_scheduler = True
        if self.runtime.max_num_steps != 1:
            logger.info(
                "Setting max-num-steps=1 for device graph capture with overlap scheduling."
            )
        self.runtime.max_num_steps = 1

    def _validate_and_resolve_max_num_steps(self) -> None:
        """Validates and resolves the max_num_steps field (platform-specific)."""
        if self.draft_model is not None and self.runtime.max_num_steps > 1:
            raise ValueError(
                f"max_num_steps must be 1 when speculative decoding is enabled, "
                f"got {self.runtime.max_num_steps}."
            )
        if self.runtime.max_num_steps < 0:
            if self.model.default_device_spec == DeviceSpec.cpu():
                self.runtime.max_num_steps = 1
            elif self.draft_model is not None:
                # Speculative decoding pipelines manage multi-step KV
                # allocation internally.
                self.runtime.max_num_steps = 1
            else:
                self.runtime.max_num_steps = 10

    def _validate_pipeline_config_for_speculative_decoding(self) -> None:
        """Validates pipeline config when used in speculative decoding mode."""
        assert self.draft_model is not None
        assert self.speculative is not None

        # Validate that both the `draft_model` and target model `model_path` have the same
        # architecture
        draft_arch_name = self.draft_model.architecture_name
        if draft_arch_name is None:
            raise ValueError(
                f"Cannot determine architecture for draft model "
                f"'{self.draft_model.model_path}': "
                "no 'architectures' field in HuggingFace config."
            )
        draft_arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=draft_arch_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
        )

        if not draft_arch:
            # Check if an eager (ModuleV3) variant exists when the graph API lookup failed
            if not self.runtime.prefer_module_v3:
                v3_arch = PIPELINE_REGISTRY.retrieve_architecture(
                    architecture_name=draft_arch_name,
                    prefer_module_v3=True,
                )
                if v3_arch:
                    raise ValueError(
                        f"MAX-optimized architecture found for draft model '{self.draft_model.model_path}', "
                        f"but only the new Module-based implementation is available (architecture: '{v3_arch.name}'). "
                        f"Please use the '--prefer-module-v3' flag to use the new implementation."
                    )
            raise ValueError(
                "MAX-Optimized architecture not found for `draft_model`"
            )

        target_arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=self.models.main_architecture_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
        )
        if not target_arch:
            # Check if an eager (ModuleV3) variant exists when the graph API lookup failed
            if not self.runtime.prefer_module_v3:
                v3_arch = PIPELINE_REGISTRY.retrieve_architecture(
                    architecture_name=self.models.main_architecture_name,
                    prefer_module_v3=True,
                )
                if v3_arch:
                    raise ValueError(
                        f"MAX-optimized architecture found for target model '{self.model.model_path}', "
                        f"but only the new Module-based implementation is available (architecture: '{v3_arch.name}'). "
                        f"Please use the '--prefer-module-v3' flag to use the new implementation."
                    )
            raise ValueError(
                "MAX-Optimized architecture not found for target model (`model_path`)"
            )

        # Validate that their tokenizers are identical.
        if self.speculative.is_standalone():
            if draft_arch != target_arch:
                raise ValueError(
                    f"architecture for the draft_model ({draft_arch.name}) does not match the architecture retrieved for the target model ({target_arch.name})"
                )

            draft_tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
                huggingface_repo=self.draft_model.huggingface_model_repo
            )
            target_tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
                huggingface_repo=self.model.huggingface_model_repo
            )

            # Compare Vocabularies
            if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
                raise ValueError(
                    f"tokenizer for draft_model ({self.draft_model.model_path}) does not match the vocabulary of the tokenizer for the target model ({self.model.model_path})"
                )

            # Compare Tokenizer Configuration
            if hasattr(draft_tokenizer, "_tokenizer") and hasattr(
                target_tokenizer, "_tokenizer"
            ):
                if (
                    draft_tokenizer._tokenizer.__dict__
                    != target_tokenizer._tokenizer.__dict__
                ):
                    raise ValueError(
                        f"tokenizer for draft_model ({self.draft_model.model_path}) does not match the configuration of the tokenizer for the target model ({self.model.model_path})"
                    )
            else:
                if draft_tokenizer.__dict__ != target_tokenizer.__dict__:
                    raise ValueError(
                        f"tokenizer for draft_model ({self.draft_model.model_path}) does not match the configuration of the tokenizer for the target model ({self.model.model_path})"
                    )

        if self.model.enable_echo:
            raise ValueError(
                "enable_echo not currently supported with speculative decoding enabled"
            )

    def _validate_and_resolve_architecture(
        self, model_config: MAXModelConfig
    ) -> SupportedArchitecture:
        """Validates and resolves architecture, quantization, rope, and encoding.

        This performs all validation up to (but not including) memory
        estimation. Returns the resolved SupportedArchitecture.
        """
        # Retrieve the architecture
        arch_name = model_config.architecture_name
        if arch_name is None:
            raise ValueError(
                f"Cannot determine architecture for '{model_config.model_path}': "
                "no 'architectures' field in HuggingFace config."
            )
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=arch_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
        )

        # If nothing is provided, we should not update any more params.
        if not arch:
            # Check if an eager (ModuleV3) variant exists when the graph API lookup failed
            if not self.runtime.prefer_module_v3:
                v3_arch = PIPELINE_REGISTRY.retrieve_architecture(
                    architecture_name=arch_name,
                    prefer_module_v3=True,
                )
                if v3_arch:
                    raise ValueError(
                        f"MAX-optimized architecture found for '{model_config.model_path}', "
                        f"but only the new Module-based implementation is available (architecture: '{v3_arch.name}'). "
                        f"Please use the '--prefer-module-v3' flag to use the new implementation.\n"
                        f"Example: max serve --model-path {model_config.model_path} --prefer-module-v3"
                    )

            raise ValueError(
                f"MAX-optimized architecture not available for '{model_config.model_path}'. "
                "Please file a request at https://modul.ar/request to add this model architecture to MAX."
            )

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

        devices = load_devices(model_config.device_specs)

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
            if len(devices) > 1:
                raise ValueError(
                    "LoRA is currently not supported with the number of devices > 1."
                )

        model_config.validate_multi_gpu_supported(
            multi_gpu_supported=arch.multi_gpu_supported
        )

        # We have now made sure that we have a valid SupportedArchitecture.
        # We should then validate the details of the existing architecture and
        # fallback to HuggingFace if needed.
        model_config.validate_and_resolve_quantization_encoding_weight_path(
            default_encoding=arch.default_encoding
        )

        # The quantization encoding has been resolved at this point.
        # This means that a KV cache dtype can be determined, assuming an override wasn't provided.
        model_config.set_cache_dtype_given_quantization_encoding()

        model_config.validate_and_resolve_rope_type(
            arch_rope_type=arch.rope_type
        )

        # by this point, the quantization_encoding must be provided. verify it is supported.
        if model_config.quantization_encoding not in arch.supported_encodings:
            raise ValueError(
                f"quantization_encoding of '{model_config.quantization_encoding}' not supported by MAX engine."
            )
        model_config.validate_and_resolve_with_resolved_quantization_encoding(
            supported_encodings=arch.supported_encodings,
            default_weights_format=arch.default_weights_format,
        )

        return arch

    def _validate_and_resolve_speculative_memory(self) -> None:
        """Memory estimation for unified speculative decoding.

        The draft model shares almost all weights with the target, so
        memory estimation uses the target model's weight size directly.
        If a future speculative method introduces a draft with significant
        non-shared weights, draft weight reservation should be added here.
        """
        assert self.draft_model is not None

        target_arch = self._validate_and_resolve_architecture(self.model)

        # Note: quantization_encoding is NOT inherited from the target model.
        # Draft models (especially EAGLE3) typically use bfloat16 regardless
        # of the target model's quantization. The draft model auto-detects
        # its encoding from its weights during architecture resolution.

        draft_arch = self._validate_and_resolve_architecture(self.draft_model)

        self._validate_and_resolve_remaining_pipeline_config(
            model_config=self.model,
            resolved_arch=target_arch,
        )

        if self.draft_model.kv_cache._available_cache_memory is not None:
            raise ValueError(
                "Expected draft model's available_cache_memory to be None"
            )
        self.draft_model.kv_cache._available_cache_memory = 0

        # Clamp max_length to the draft model's max sequence length.
        # EAGLE and other draft models may support a shorter context than the
        # target model (e.g. 2048 vs 131072).  Both models share a KV cache
        # and must agree on the sequence length, so we use the minimum.
        draft_arch_config = draft_arch.config.initialize(
            self, model_config=self.draft_model
        )
        draft_max_seq_len = draft_arch_config.get_max_seq_len()
        target_max_length = self.model.max_length
        if (
            target_max_length is not None
            and target_max_length > draft_max_seq_len
        ):
            logger.info(
                f"Clamping max_length from {target_max_length} to"
                f" {draft_max_seq_len} (draft model max sequence length)"
            )
            self.model.max_length = draft_max_seq_len
            self.draft_model.max_length = draft_max_seq_len

    def _validate_and_resolve_remaining_pipeline_config(
        self,
        model_config: MAXModelConfig,
        resolved_arch: SupportedArchitecture | None = None,
    ) -> None:
        """Validates remaining config fields and runs memory estimation.

        Args:
            model_config: The model configuration to validate and resolve.
            resolved_arch: Pre-resolved architecture, skips re-validation.
        """
        arch = resolved_arch or self._validate_and_resolve_architecture(
            model_config
        )

        if is_diffusion_pipeline(model_config.huggingface_model_repo):
            # Skip memory estimation for diffusion pipelines,
            # since they don't use KV cache.
            return

        if not issubclass(arch.pipeline_model, PipelineModel):
            # Non-PipelineModel architectures (e.g. PipelineExecutor) skip
            # memory estimation — they don't expose these classmethods.
            return

        devices = load_devices(model_config.device_specs)
        arch_config = arch.config.initialize(self, model_config=model_config)

        weights_size = arch.pipeline_model.estimate_weights_size(self)
        activation_size = arch.pipeline_model.estimate_activation_memory(
            self, model_config.huggingface_config
        )

        MemoryEstimator.estimate_memory_footprint(
            self,
            model_config,
            arch_config,
            devices,
            weights_size,
            activation_size,
        )

        if clamped_max_seq_len := MemoryEstimator.max_supported_sequence_length(
            weights_size,
            activation_size,
            model_config,
            devices,
            arch_config,
        ):
            if self.model.max_length is None:
                self.model.max_length = clamped_max_seq_len
            elif self.model.max_length > clamped_max_seq_len:
                logging.warning(
                    f"Clamping max_length from {self.model.max_length} to {clamped_max_seq_len} due to capacity of KV Cache"
                )
                self.model.max_length = clamped_max_seq_len

        # Validate whether the architecture requires a max batch total tokens to be specified.
        # This needs to be done after max_length is resolved.
        if (
            arch.requires_max_batch_context_length
            and self.runtime.max_batch_total_tokens is None
        ):
            logger.warning(
                f"Architecture '{arch.name}' requires max-batch-total-tokens to be specified but found None. "
                f"Defaulting to the max sequence length of the model: {self.model.max_length}"
            )
            self.runtime.max_batch_total_tokens = self.model.max_length

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

    @property
    def graph_quantization_encoding(self) -> QuantizationEncoding | None:
        """Converts the CLI encoding to a MAX graph quantization encoding.

        Returns:
            The graph quantization encoding corresponding to the CLI encoding.
        """
        return self.model.graph_quantization_encoding

    def log_pipeline_info(self) -> None:
        """Logs comprehensive pipeline and KVCache configuration information.

        Retrieves all necessary information from self and the PIPELINE_REGISTRY.
        Raises an error if architecture is not found (which should not happen after config resolution).
        """
        # Retrieve architecture - this should always exist after config resolution
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=self.models.main_architecture_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
        )

        if arch is None:
            raise ValueError(
                f"No architecture found for {self.models.main_architecture_name}. "
                "This should not happen after config resolution."
            )

        # Get pipeline task and class information
        pipeline_class = get_pipeline_for_task(arch.task, self)

        # Log architecture and pipeline class information
        arch_entries: list[tuple[str, Any]] = [
            ("architecture", arch.name),
            ("pipeline_class", pipeline_class.__name__),
            ("pipeline_model", arch.pipeline_model.__name__),
            ("tokenizer", arch.tokenizer_cls.__name__),
        ]

        logger.info("")
        logger.info("Pipeline Architecture")
        logger.info("=" * 60)
        for line in _format_config_entries(arch_entries):
            logger.info(line)

        # Delegate model-specific logging to the manifest
        self.models.log_model_info()
        pipeline_entries: list[tuple[str, Any]] = []
        if "main" in self.models:
            pipeline_entries.append(("max_seq_len", self.model.max_length))
        pipeline_entries.extend(
            [
                ("max_batch_size", self.runtime.max_batch_size),
                ("chunked_prefill", self.runtime.enable_chunked_prefill),
                ("max_batch_input_tokens", self.runtime.max_batch_input_tokens),
                (
                    "in_flight_batching",
                    self.runtime.enable_in_flight_batching,
                ),
            ]
        )

        logger.info("")
        logger.info("Pipeline Config")
        logger.info("=" * 60)
        for line in _format_config_entries(pipeline_entries):
            logger.info(line)
        logger.info("")

        # Denoising cache details for diffusion pipelines.
        if arch.task == PipelineTask.PIXEL_GENERATION:
            cache = self.runtime.denoising_cache
            cache_entries: list[tuple[str, Any]] = [
                ("first_block_caching", cache.first_block_caching),
                ("taylorseer", cache.taylorseer),
                (
                    "taylorseer_cache_interval",
                    cache.taylorseer_cache_interval
                    if cache.taylorseer_cache_interval is not None
                    else "model-default",
                ),
                (
                    "taylorseer_warmup_steps",
                    cache.taylorseer_warmup_steps
                    if cache.taylorseer_warmup_steps is not None
                    else "model-default",
                ),
                (
                    "taylorseer_max_order",
                    cache.taylorseer_max_order
                    if cache.taylorseer_max_order is not None
                    else "model-default",
                ),
                ("teacache", cache.teacache),
                (
                    "teacache_rel_l1_thresh",
                    cache.teacache_rel_l1_thresh
                    if cache.teacache_rel_l1_thresh is not None
                    else "model-default",
                ),
                (
                    "teacache_coefficients",
                    cache.teacache_coefficients
                    if cache.teacache_coefficients is not None
                    else "model-default",
                ),
            ]

            logger.info("Denoising Cache")
            logger.info("=" * 60)
            for line in _format_config_entries(cache_entries):
                logger.info(line)
            logger.info("")

    def log_basic_config(self) -> None:
        """Log minimal pipeline configuration information.

        Logs basic :class:`~max.pipelines.lib.config.PipelineConfig` options including model name, pipeline task,
        weight path, max_batch_size, max_seq_len, and reserved memory.
        """
        # Retrieve architecture - this should always exist after config resolution
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            architecture_name=self.models.main_architecture_name,
            prefer_module_v3=self.runtime.prefer_module_v3,
        )

        if arch is None:
            model_path = (
                self.models.main_architecture_name
                if "main" in self.models
                else str(list(self.models.keys()))
            )
            raise ValueError(
                f"No architecture found for {model_path}. "
                "This should not happen after config resolution."
            )

        task = arch.task
        pipeline_class = get_pipeline_for_task(task, self)

        # Get reserved memory info from KVCache config (only for tasks that use KV cache)
        kv_cache_tasks = {
            PipelineTask.TEXT_GENERATION,
            PipelineTask.AUDIO_GENERATION,
            PipelineTask.SPEECH_TOKEN_GENERATION,
        }

        memory_str = None
        if "main" in self.models and task in kv_cache_tasks:
            kv_config = self.model.kv_cache
            if kv_config._available_cache_memory is None:
                raise ValueError(
                    "KVCache config is not available after config resolution."
                )
            memory_str = to_human_readable_bytes(
                kv_config._available_cache_memory
            )

        # Log basic configuration
        config_entries: list[tuple[str, Any]] = [
            ("architecture", arch.name),
            ("pipeline", pipeline_class.__name__),
        ]
        if "main" in self.models:
            devices_str = ", ".join(
                f"{d.device_type}[{d.id}]" for d in self.model.device_specs
            )
            config_entries.extend(
                [
                    ("model", self.model.model_path),
                    ("devices", devices_str),
                    ("max_batch_size", self.runtime.max_batch_size),
                    ("max_seq_len", self.model.max_length),
                ]
            )
        else:
            config_entries.append(
                ("max_batch_size", self.runtime.max_batch_size)
            )

        if memory_str:
            config_entries.append(("cache_memory", memory_str))
        config_entries.append(
            ("device_graph_capture", self.runtime.device_graph_capture)
        )

        if self.speculative is not None:
            config_entries.append(
                ("speculative_method", self.speculative.speculative_method)
            )
            config_entries.append(
                (
                    "num_speculative_tokens",
                    self.speculative.num_speculative_tokens,
                )
            )
            if self.speculative.use_relaxed_acceptance_for_thinking:
                config_entries.append(
                    ("relaxed_topk", self.speculative.relaxed_topk)
                )
                config_entries.append(
                    ("relaxed_delta", self.speculative.relaxed_delta)
                )

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "Pipeline Configuration (use --pretty-print-config to print full config)"
        )
        logger.info("=" * 60)
        for line in _format_config_entries(config_entries):
            logger.info(line)
        logger.info("")


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


PrependPromptSpeechTokens = Literal["never", "once", "rolling"]
"""Controls whether prompt speech tokens are prepended to the audio decoder.

``"never"``
    Never prepend the prompt speech tokens sent to the audio decoder.
``"once"``
    Prepend the prompt speech tokens to the first block of the audio decoder.
``"rolling"``
    Prepend the prompt speech tokens to the first block of the audio decoder,
    and to later blocks to reach the requested buffer size.
"""


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


class AudioGenerationConfig(PipelineConfig):
    """Configuration for an audio generation pipeline."""

    # TODO: Make these flags more discoverable.
    audio_decoder: str = Field(
        default="",
        description="The name of the audio decoder model architecture.",
    )
    """The name of the audio decoder model architecture."""

    audio_decoder_weights: str = Field(
        default="", description="The path to the audio decoder weights file."
    )
    """The path to the audio decoder weights file."""

    chunk_size: list[int] | None = Field(
        default=None,
        description=(
            "The chunk sizes to use for streaming. If this is an int, fixed-size "
            "chunks of the given size are used. If this is a list, variable "
            "chunk sizes are used."
        ),
    )
    """The chunk sizes to use for streaming."""

    buffer: int = Field(
        default=0,
        description=(
            "The number of previous speech tokens to pass to the audio decoder "
            "on each generation step."
        ),
    )
    """The number of previous speech tokens to pass to the audio decoder on each generation step."""

    block_causal: bool = Field(
        default=False,
        description=(
            "Whether prior buffered tokens should attend to tokens in the "
            "current block. Has no effect if buffer is not set."
        ),
    )
    """Whether prior buffered tokens attend to tokens in the current block."""

    prepend_prompt_speech_tokens: PrependPromptSpeechTokens = Field(
        default="once",
        description=(
            "Whether the prompt speech tokens should be forwarded to the audio "
            "decoder. Options: never, once, rolling."
        ),
    )
    """Whether the prompt speech tokens are forwarded to the audio decoder."""

    prepend_prompt_speech_tokens_causal: bool = Field(
        default=False,
        description=(
            "Whether the prompt speech tokens should attend to tokens in the "
            "currently generated audio block. Has no effect if "
            "prepend_prompt_speech_tokens is never."
        ),
    )
    """Whether the prompt speech tokens attend to tokens in the current audio block."""

    audio_decoder_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the audio decoder model.",
    )
    """Parameters to pass to the audio decoder model."""

    prometheus_metrics_mode: PrometheusMetricsMode = Field(
        default="instrument_only",
        description="The mode to use for Prometheus metrics.",
    )
    """The mode to use for Prometheus metrics."""

    _run_model_test_mode: bool = PrivateAttr(default=False)
    """Test-only flag that indicates that test parameters have been passed to
    the model, such as leaving the audio decoder weights empty or using a
    dummy speech language model."""

    def __init__(
        self,
        audio_decoder: str,
        audio_decoder_weights: str = "",
        chunk_size: list[int] | None = None,
        buffer: int = 0,
        block_causal: bool = False,
        prepend_prompt_speech_tokens: PrependPromptSpeechTokens = "never",
        prepend_prompt_speech_tokens_causal: bool = False,
        run_model_test_mode: bool = False,
        prometheus_metrics_mode: PrometheusMetricsMode = "instrument_only",
        **kwargs: Any,
    ) -> None:
        # Must call the superclass's __init__ first, otherwise PipelineConfig's
        # init will override values defined in the AudioGenerationConfig.
        PipelineConfig.__init__(self, **kwargs)
        if block_causal:
            raise NotImplementedError("Causal generation is not implemented")
        if prepend_prompt_speech_tokens_causal:
            raise NotImplementedError(
                "Prepend prompt speech tokens causal is not implemented"
            )

        self.audio_decoder = audio_decoder
        self.audio_decoder_weights = audio_decoder_weights
        self.chunk_size = chunk_size
        self.buffer = buffer
        self.block_causal = block_causal
        self.prepend_prompt_speech_tokens = prepend_prompt_speech_tokens
        self.prepend_prompt_speech_tokens_causal = (
            prepend_prompt_speech_tokens_causal
        )
        self._run_model_test_mode = run_model_test_mode
        self.prometheus_metrics_mode = prometheus_metrics_mode

    @classmethod
    def from_flags(
        cls, audio_flags: dict[str, str], **config_flags: Any
    ) -> AudioGenerationConfig:
        """Builds an :class:`~max.pipelines.lib.config.AudioGenerationConfig` from audio CLI flags and config kwargs."""
        audio_decoder = audio_flags.pop("audio_decoder", "")
        if not audio_decoder:
            raise ValueError(
                "When running the audio generation task, --audio-decoder must be specified"
            )
        audio_decoder_weights = audio_flags.pop("audio_decoder_weights", "")

        # Configuration for audio generation streaming.
        chunk_size_str = audio_flags.pop("chunk_size", "")
        if not chunk_size_str:
            chunk_size = None
        else:
            chunk_size = [int(size) for size in chunk_size_str.split(",")]

        buffer = _parse_flag_int(audio_flags.pop("buffer", "0"), "buffer")

        block_causal = _parse_flag_bool(
            audio_flags.pop("block_causal", "false"), "block_causal"
        )

        prepend_prompt_speech_tokens = cast(
            PrependPromptSpeechTokens,
            audio_flags.pop("prepend_prompt_speech_tokens", "never"),
        )

        prepend_prompt_speech_tokens_causal = _parse_flag_bool(
            audio_flags.pop("prepend_prompt_speech_tokens_causal", "false"),
            "prepend_prompt_speech_tokens_causal",
        )

        run_model_test_mode = _parse_flag_bool(
            audio_flags.pop("run_model_test_mode", "false"),
            "run_model_test_mode",
        )

        prometheus_metrics_mode = cast(
            PrometheusMetricsMode,
            audio_flags.pop("prometheus_metrics_mode", "instrument_only"),
        )

        if audio_flags:
            raise ValueError(
                f"Unknown audio generation option(s): {audio_flags}"
            )

        return cls(
            audio_decoder=audio_decoder,
            audio_decoder_weights=audio_decoder_weights,
            chunk_size=chunk_size,
            buffer=buffer,
            block_causal=block_causal,
            prepend_prompt_speech_tokens=prepend_prompt_speech_tokens,
            prepend_prompt_speech_tokens_causal=prepend_prompt_speech_tokens_causal,
            run_model_test_mode=run_model_test_mode,
            prometheus_metrics_mode=prometheus_metrics_mode,
            **config_flags,
        )

    @override
    def _validate_and_resolve_overlap_scheduler(self) -> None:
        if self.runtime.force:
            return

        if self.runtime.enable_overlap_scheduler:
            raise ValueError(
                "The Overlap scheduler does not support Audio Generation. "
                "Detected AudioGenerationConfig."
            )
