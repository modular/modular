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

import json
import logging
import tempfile
from typing import TYPE_CHECKING, Any, Literal, get_args

from max.config import ConfigFileModel
from max.driver import accelerator_api, load_devices
from max.engine import InferenceSession
from max.graph.quantization import QuantizationEncoding
from max.nn.comm import Signals
from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.diffusion.cache import DenoisingCacheConfig
from max.pipelines.kv_cache.config import KVCacheConfig, KVConnectorConfig
from max.pipelines.lib.interfaces import (
    ArchConfig,
    ArchConfigWithKVCache,
    PipelineModel,
)
from max.pipelines.lib.memory_estimation import (
    MemoryEstimator,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import (
    DISABLE_PARSER_SENTINEL,
    PipelineRuntimeConfig,
)
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.types.task import PipelineTask
from max.pipelines.sampling import SamplingConfig
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.weights.hf_utils import is_diffusion_pipeline
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_validator,
)
from typing_extensions import Self

from .model_config import MAXModelConfig
from .profiling_config import ProfilingConfig

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

        Drives whether model / sampler graphs are compiled with a bitmask
        input and whether the D2H pinned buffer is allocated. Distinct from
        ``sampling.enable_structured_output``, which is the user-facing
        flag and only gates honoring user-supplied JSON schemas.
        """
        return (
            self.sampling.enable_structured_output
            or self.runtime.tool_parser is not None
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
            elif "main" in self.models:
                # The main model came from a YAML recipe (or a pre-built
                # manifest via ``models=``). Still let CLI flags such as
                # --devices override the recipe so the same YAML can be
                # reused across different multi-GPU setups.
                non_default_kwargs = _strip_default_model_kwargs(model_kwargs)
                if non_default_kwargs:
                    self.models = self.models.with_override(
                        "main", **non_default_kwargs
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

            if "main" in self.models and self.model.enable_echo:
                sampling_config.enable_variable_logits = True
            setattr(self, config_name, sampling_config)
        else:
            existing = getattr(self, config_name, None)
            if existing is not None and isinstance(existing, config_class):
                merged = existing.model_copy(update=matched_kwargs)
                setattr(self, config_name, merged)
            else:
                setattr(self, config_name, config_class(**matched_kwargs))

    @classmethod
    def from_flat_kwargs(cls, **kwargs: Any) -> Self:
        """Construct a :class:`PipelineConfig` from a flat CLI kwargs namespace.

        Accepts the flat kwargs produced by ``pipeline_config_options`` (for
        example ``model_path``, ``kv_cache_size``, ``enable_lora``) and routes
        them into the appropriate sub-configs before constructing the instance.

        This is the entry point for CLI and legacy callers. Direct construction
        via ``PipelineConfig(models=..., runtime=..., ...)`` with properly typed
        sub-configs is also supported and requires no routing.
        """
        # Merge YAML config file values before routing, then clear config_file
        # so the Pydantic model_validator on ConfigFileModel doesn't reload it
        # when cls(**pydantic_kwargs) is called below.
        kwargs = cls.load_config_file(kwargs)  # type: ignore[operator]
        kwargs.pop("config_file", None)

        # Intercept legacy model/draft_model kwargs — these are no longer
        # Pydantic fields but some callers still pass them directly.
        model_kwarg = kwargs.pop("model", None)
        draft_model_kwarg = kwargs.pop("draft_model", None)

        # If a MAXModelConfig (or plain dict from config file) was passed
        # directly, wrap it in a manifest.
        if model_kwarg is not None:
            if isinstance(model_kwarg, dict) and not isinstance(
                model_kwarg, MAXModelConfig
            ):
                model_kwarg = MAXModelConfig(**model_kwarg)
            kwargs["models"] = ModelManifest({"main": model_kwarg})

        # Separate PipelineConfig-own fields from sub-config fields.
        pydantic_kwargs: dict[str, Any] = {}
        unmatched_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in cls.model_fields:
                pydantic_kwargs[key] = value
            else:
                unmatched_kwargs[key] = value

        instance = cls(**pydantic_kwargs)

        # Add draft model via with_override.
        if draft_model_kwarg is not None:
            if isinstance(draft_model_kwarg, dict) and not isinstance(
                draft_model_kwarg, MAXModelConfig
            ):
                draft_model_kwarg = MAXModelConfig(**draft_model_kwarg)
            instance.models = instance.models.with_override(
                "draft", config=draft_model_kwarg
            )

        # Route unmatched kwargs into sub-configs.  Ordering matters:
        # - models before sampling (sampling needs generation_config from model)
        # - models before speculative (speculative needs draft_model)
        # - runtime before denoising_cache (denoising_cache is set on runtime)
        instance._create_lora_config_if_needed(unmatched_kwargs)
        instance._build_models_from_kwargs(unmatched_kwargs)
        instance._create_speculative_config_if_needed(unmatched_kwargs)
        if unmatched_kwargs:
            instance._process_remaining_config_classes(unmatched_kwargs)
        instance._create_denoising_cache_config_if_needed(unmatched_kwargs)

        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        return instance

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
        if target_archs[0] == "MiniMaxM3SparseForConditionalGeneration":
            if self.speculative.is_mtp() and self.draft_model is None:
                target_archs[0] = (
                    "UnifiedMTPMiniMaxM3SparseForConditionalGeneration"
                )

    def resolve(
        self,
        arch: Any,
        draft_arch: Any = None,
    ) -> None:
        """Validates and resolves the config.

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

        # NOTE: the unified spec-decode target-architecture override
        # (``_resolve_speculative_target_architecture``) is applied by the
        # registry *before* it resolves ``arch`` and passes it in here, so that
        # the ``arch`` consumed by memory estimation, the overlap scheduler, and
        # parser resolution below already reflects the override. Applying it
        # here (after ``arch`` is resolved) would leave those consumers using
        # the stale pre-override architecture. See SERVOPT regression from
        # PipelineConfig/registry decoupling (#88511).

        # Validate KV connector configuration
        _resolve_kvconnector_config(self.model.kv_cache)

        # By this point, we should have a valid model_path.

        if self.draft_model:
            # Joint memory estimation for speculative decoding
            _resolve_kvconnector_config(self.draft_model.kv_cache)
            self._validate_and_resolve_speculative_memory(
                target_arch=arch, draft_arch=draft_arch
            )
            self._validate_pipeline_config_for_speculative_decoding(
                target_arch=arch,
                draft_arch=draft_arch,
            )
        else:
            self._validate_and_resolve_remaining_pipeline_config(
                model_config=self.model, resolved_arch=arch
            )

        self._validate_and_resolve_overlap_scheduler(arch=arch)

        self._resolve_default_reasoning_parser(arch=arch)
        self._resolve_default_tool_parser(arch=arch)

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

    def _validate_and_resolve_overlap_scheduler(self, arch: Any = None) -> None:
        if not self.runtime.force:
            max_batch_size = self.runtime.max_batch_size
            if (
                self.runtime.device_graph_capture is None
                and arch is not None
                and arch.supports_device_graph_capture
                and max_batch_size is not None
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
            if self.model.device_specs[0].device_type == "cpu":
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

    def _validate_and_resolve_max_num_steps(self) -> None:
        """Normalize deprecated ``max_num_steps`` to single-step decode."""
        if self.runtime.max_num_steps in (1, -1):
            self.runtime.max_num_steps = 1
            return

        logger.warning(
            "--max-num-steps=%s is deprecated and ignored; using single-step "
            "decode (max_num_steps=1).",
            self.runtime.max_num_steps,
        )
        self.runtime.max_num_steps = 1

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

    def _validate_and_resolve_speculative_memory(
        self, target_arch: Any, draft_arch: Any
    ) -> None:
        """Memory estimation for unified speculative decoding.

        The draft model shares almost all weights with the target, so
        memory estimation uses the target model's weight size directly.
        If a future speculative method introduces a draft with significant
        non-shared weights, draft weight reservation should be added here.

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
        # _validate_and_resolve_remaining_pipeline_config below.
        self._validate_model_config_against_arch(self.draft_model, draft_arch)

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
        resolved_arch: Any,
    ) -> None:
        """Validates model config against the architecture and runs memory estimation.

        Args:
            model_config: The model configuration to validate and resolve.
            resolved_arch: Pre-resolved architecture from the registry.
        """
        self._validate_model_config_against_arch(model_config, resolved_arch)

        if is_diffusion_pipeline(model_config.huggingface_model_repo):
            # Skip memory estimation for diffusion pipelines,
            # since they don't use KV cache.
            return

        if not issubclass(resolved_arch.pipeline_model, PipelineModel):
            # Non-PipelineModel architectures (e.g. PipelineExecutor) skip
            # memory estimation.
            return

        devices = load_devices(model_config.device_specs)
        arch_config = resolved_arch.config.initialize(
            self, model_config=model_config
        )

        if resolved_arch.memory_planner is not None:
            planner = resolved_arch.memory_planner(arch_config)
            weights_size = planner.estimate_weights_size(self)
            activation_size = planner.estimate_activation_memory(
                self, model_config.huggingface_config
            )
            signal_buffer_size = planner.estimate_signal_buffer_memory(
                self, arch_config
            )
        else:
            # ``memory_planner=None`` is the intentional state for architectures
            # that manage memory outside the planner path (e.g. diffusion models,
            # which exit early via ``is_diffusion_pipeline()`` before reaching
            # this point).  It is also the fallback for any architecture not yet
            # wired to a MemoryPlanner — if you are adding a new architecture
            # that uses a KV cache, set ``memory_planner=PagedMemoryPlanner``
            # on your ``SupportedArchitecture`` to get correct memory estimation.
            weights_size = model_config.weights_size()
            activation_size = 0
            signal_buffer_size = self.estimate_signal_buffer_memory(arch_config)

        MemoryEstimator.estimate_memory_footprint(
            self,
            model_config,
            arch_config,
            devices,
            weights_size,
            activation_size,
            signal_buffer_size,
            arch=resolved_arch,
        )

        if clamped_max_seq_len := MemoryEstimator.max_supported_sequence_length(
            weights_size,
            activation_size,
            model_config,
            devices,
            arch_config,
            signal_buffer_size,
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
            resolved_arch.requires_max_batch_context_length
            and self.runtime.max_batch_total_tokens is None
        ):
            logger.warning(
                f"Architecture '{resolved_arch.name}' requires max-batch-total-tokens to be specified but found None. "
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
