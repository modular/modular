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
from enum import Enum
from pathlib import Path
from typing import Any, get_type_hints

from max.config import ConfigFileModel
from max.driver import DeviceSpec, load_devices
from max.engine import InferenceSession
from max.graph.quantization import QuantizationEncoding
from max.serve.worker_interface.zmq_queue import generate_zmq_ipc_path
from pydantic import (
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self, override

from .config_enums import PipelineRole
from .hf_utils import is_diffusion_pipeline
from .kv_cache_config import KVCacheConfig
from .lora_config import LoRAConfig
from .memory_estimation import MemoryEstimator, to_human_readable_bytes
from .model_config import MAXModelConfig
from .profiling_config import ProfilingConfig
from .registry import (
    PIPELINE_REGISTRY,
    SupportedArchitecture,
    get_pipeline_for_task,
)
from .sampling import SamplingConfig
from .speculative_config import SpeculativeConfig

logger = logging.getLogger("max.pipelines")

# Default max batch input tokens for chunked prefill and memory estimation.
DEFAULT_MAX_BATCH_INPUT_TOKENS = 8192


class PipelineConfig(ConfigFileModel):
    """Configuration for a pipeline.

    WIP - Once a PipelineConfig is fully initialized, it should be as immutable
    as possible (frozen=True). All underlying dataclass fields should have been
    initialized to their default values, be it user specified via some CLI
    flag, config file, environment variable, or internally set to a reasonable
    default.
    """

    max_length: int | None = Field(
        default=None, description="Maximum sequence length of the model."
    )

    pipeline_role: PipelineRole = Field(
        default=PipelineRole.PrefillAndDecode,
        description=(
            "Whether the pipeline should serve both a prefill or decode role or "
            "both."
        ),
    )

    max_batch_size: int | None = Field(
        default=None,
        description=(
            "Maximum batch size to execute with the model. When not specified "
            "(None), this value is determined dynamically. For server launches, "
            "set this higher based on server capacity. When "
            "device_graph_capture is enabled, overlap pre-captures decode "
            "graph entries for batch sizes [1..max_batch_size]."
        ),
    )

    max_queue_size_tg: int | None = Field(
        default=None,
        description=(
            "Maximum number of requests in decode queue. By default, this is "
            "max_batch_size."
        ),
    )

    min_batch_size_tg: int | None = Field(
        default=None,
        description=(
            "Soft floor on the decode batch size. If the TG batch size is "
            "larger, the scheduler continues TG batches; if it falls below, the "
            "scheduler prioritizes CE. This is not a strict minimum. By "
            "default, this is max_queue_size_tg. Experimental for the TTS "
            "scheduler."
        ),
    )

    ep_size: int = Field(
        default=1,
        description=(
            "The expert parallelism size. Needs to be 1 (no expert parallelism) "
            "or the total number of GPUs across nodes."
        ),
    )

    ce_delay_ms: float = Field(
        default=0.0,
        description=(
            "Duration of scheduler sleep prior to starting a prefill batch. "
            "Experimental for the TTS scheduler."
        ),
    )

    enable_prioritize_first_decode: bool = Field(
        default=False,
        description=(
            "When enabled, the scheduler always runs a TG batch immediately "
            "after a CE batch with the same requests. This may reduce "
            "time-to-first-chunk latency. Experimental for the TTS scheduler."
        ),
    )

    enable_chunked_prefill: bool = Field(
        default=True,
        description=(
            "Enable chunked prefill to split context encoding requests into "
            "multiple chunks based on max_batch_input_tokens."
        ),
    )

    enable_in_flight_batching: bool = Field(
        default=False,
        description=(
            "When enabled, prioritizes token generation by batching it with "
            "context encoding requests."
        ),
    )

    max_num_steps: int = Field(
        default=-1,
        description=(
            "The number of steps to run for multi-step scheduling. -1 specifies "
            "a default value based on configuration and platform. Ignored for "
            "models which are not auto-regressive (e.g. embedding models)."
        ),
    )

    max_batch_input_tokens: int = Field(
        default=DEFAULT_MAX_BATCH_INPUT_TOKENS,
        description=(
            "The target number of un-encoded tokens to include in each batch. "
            "This value is used for chunked prefill and memory estimation."
        ),
    )

    enable_echo: bool = Field(
        default=False,
        description="Whether the model should be built with echo capabilities.",
    )

    pool_embeddings: bool = Field(
        default=True, description="Whether to pool embedding outputs."
    )

    chat_template: Path | None = Field(
        default=None,
        description=(
            "Optional custom chat template to override the one shipped with the "
            "Hugging Face model config. If a path is provided, the file is read "
            "during config resolution and the content stored as a string. If "
            "None, the model's default chat template is used."
        ),
    )

    use_experimental_kernels: str = Field(
        default=os.environ.get("USE_EXPERIMENTAL_KERNELS", "false"),
        description=(
            "Enables using experimental mojo kernels with max serve. The "
            "kernels could be unstable or incorrect."
        ),
    )

    use_vendor_blas: str = Field(
        default=os.environ.get("MAX_SERVE_USE_VENDOR_BLAS", "false"),
        description=(
            "Enables using vendor BLAS libraries (cublas/hipblas/etc) with max "
            "serve. Currently, this just replaces matmul calls."
        ),
    )

    pdl_level: str = Field(
        default=os.environ.get("PDL_LEVEL", "0"),
        description=(
            "Level of overlap of kernel launch via programmatic dependent grid "
            "control."
        ),
    )

    custom_architectures: list[str] = Field(
        default_factory=list,
        description=(
            "Custom architecture implementations to register. Each input can "
            "either be a raw module name or an import path followed by a colon "
            "and the module name. Each module must expose an ARCHITECTURES list "
            "of architectures to register."
        ),
    )

    zmq_endpoint_base: str = Field(
        default_factory=generate_zmq_ipc_path,
        description=(
            "Prefix for ZMQ endpoints used for IPC. This ensures unique "
            "endpoints across MAX Serve instances on the same host. Example: "
            'lora_request_zmq_endpoint = f"{zmq_endpoint_base}-lora_request".'
        ),
    )

    execute_empty_batches: bool = Field(
        default=False,
        description="Whether the scheduler should execute empty batches.",
    )

    max_batch_total_tokens: int | None = Field(
        default=None,
        description=(
            "Ensures the sum of page-aligned context lengths in a batch does "
            "not exceed max_batch_total_tokens. Alignment uses the KV cache "
            "page size. If None, the sum is not limited."
        ),
    )

    device_graph_capture: bool = Field(
        default=False,
        description="Enable device graph capture/replay for graph execution.",
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
            "Projected cache usage threshold for scheduling CE requests, "
            "considering current and incoming requests. CE is scheduled if "
            "either projected usage stays below this threshold or no active "
            "requests exist. Higher values can cause more preemptions."
        ),
    )

    enable_overlap_scheduler: bool = Field(
        default=False,
        description=(
            "Whether to enable the overlap scheduler. This feature allows the scheduler "
            "to run alongside GPU execution. This helps improve GPU utilization. "
            "This is an experimental feature which may crash and burn. "
            "This feature will be enabled by default for some selected architectures. "
            "You can forcibly disable this by setting --no-enable-overlap-scheduler --force."
        ),
    )

    use_legacy_module: bool = Field(
        default=True,
        description=(
            "Whether to prefer the legacy ModuleV2 architecture (default=True for backward "
            "compatibility). When True, tries the ModuleV2 architecture first and falls back "
            "to ModuleV3. When False, tries ModuleV3 first and falls back to ModuleV2."
        ),
    )

    # TODO(SERVSYS-1096): Remove this field once we've reworked how required
    # config fields are validated.
    defer_resolve: bool = Field(
        default=False,
        description="Whether to defer resolving the pipeline config.",
    )

    model: MAXModelConfig = Field(
        default_factory=MAXModelConfig, description="The model config."
    )

    draft_model: MAXModelConfig | None = Field(
        default=None, description="The draft model config."
    )

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, description="The sampling config."
    )

    profiling: ProfilingConfig = Field(
        default_factory=ProfilingConfig, description="The profiling config."
    )

    lora: LoRAConfig | None = Field(
        default=None, description="The LoRA config."
    )

    speculative: SpeculativeConfig | None = Field(
        default=None, description="The SpeculativeConfig."
    )

    _config_file_section_name: str = PrivateAttr(default="pipeline_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    _unmatched_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Temporary storage for unmatched kwargs during initialization.
    This is used to pass unmatched kwargs from the before validator to the after validator."""

    def configure_session(self, session: InferenceSession) -> None:
        """Configure an InferenceSession with standard pipeline settings."""
        session.gpu_profiling(self.profiling.gpu_profiling)
        session._use_experimental_kernels(self.use_experimental_kernels)
        session._use_vendor_blas(self.use_vendor_blas)
        session._pdl_level(self.pdl_level)

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
            key_prefix: Optional prefix to filter keys (e.g., "draft_")
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

    # TODO: It might be cleaner to have the draft model be a part of the SpeculativeConfig
    def _create_draft_model_config_if_needed(
        self, kwargs: dict[str, Any]
    ) -> None:
        """Extract draft model kwargs and create MAXModelConfig if model_path provided."""
        draft_kwargs = PipelineConfig._extract_kwargs_for_config(
            kwargs, MAXModelConfig, key_prefix="draft_", strip_prefix=True
        )

        if draft_kwargs.get("model_path", "") != "":
            self.draft_model = MAXModelConfig(**draft_kwargs)
        # TODO: We should add an elif to check / error out if other draft model
        # params are provided, but model_path is not. We can't do this today
        # as our click PipelineConfig autogenerates defaults for all fields,
        # including required ones.

    def _create_speculative_config_if_needed(
        self, kwargs: dict[str, Any]
    ) -> None:
        """Extract speculative config kwargs and create SpeculativeConfig if any speculative parameters provided."""
        speculative_kwargs = PipelineConfig._extract_kwargs_for_config(
            kwargs, SpeculativeConfig
        )
        # Only create speculative config if speculative_method is explicitly set
        if (
            speculative_kwargs
            and speculative_kwargs.get("speculative_method") is not None
        ):
            # Remove None values to use defaults
            filtered_kwargs = {
                k: v for k, v in speculative_kwargs.items() if v is not None
            }
            if filtered_kwargs:
                self.speculative = SpeculativeConfig(**filtered_kwargs)
                assert self.draft_model is not None
                # We need to set the architecture to LlamaForCausalLMEagle for Eagle speculative decoding
                if self.speculative.is_eagle():
                    if self.draft_model.huggingface_config is None:
                        raise ValueError(
                            f"EAGLE speculative decoding requires a HuggingFace config for the draft model, "
                            f"but could not load config for '{self.draft_model.model_path}'. "
                            "Please ensure the draft model is a standard Transformers model with a valid config.json."
                        )
                    if (
                        len(self.draft_model.huggingface_config.architectures)
                        != 1
                    ):
                        raise ValueError(
                            f"Expected exactly 1 architecture in draft model config, "
                            f"got {len(self.draft_model.huggingface_config.architectures)}"
                        )
                    hf_arch = self.draft_model.huggingface_config.architectures[
                        0
                    ]
                    if hf_arch == "LlamaForCausalLM":
                        self.draft_model.huggingface_config.architectures[0] = (
                            "LlamaForCausalLMEagle"
                        )

    def _process_remaining_config_classes(
        self, unmatched_kwargs: dict[str, Any]
    ) -> None:
        """Processes remaining kwargs for other config classes.

        Args:
            unmatched_kwargs: Dictionary of kwargs that haven't been matched yet
        """
        # TODO(zheng): Make this more efficient by using MaxConfig instance
        # instead of hardcoding the config names.
        config_mappings = [
            # NOTE: model must come before sampling so that
            # SamplingConfig can use generation_config from the model
            "model",
            "sampling",
            "profiling",
        ]

        for config_name in config_mappings:
            config_class = get_type_hints(self.__class__)[config_name]
            matched_kwargs = {}
            kv_cache_kwargs = {}

            for key, value in unmatched_kwargs.items():
                if key in config_class.model_fields:
                    matched_kwargs[key] = value
                # Check if this is a KVCache config param
                elif (
                    config_name == "model" and key in KVCacheConfig.model_fields
                ):
                    kv_cache_kwargs[key] = value

            if matched_kwargs:
                self._create_and_set_config(
                    config_name, config_class, matched_kwargs, kv_cache_kwargs
                )

                # Remove matched kwargs
                for key in matched_kwargs:
                    _ = unmatched_kwargs.pop(key, None)
                for key in kv_cache_kwargs:
                    _ = unmatched_kwargs.pop(key, None)

    def _create_and_set_config(
        self,
        config_name: str,
        config_class: type,
        matched_kwargs: dict[str, Any],
        kv_cache_kwargs: dict[str, Any],
    ) -> None:
        """Creates and sets a config object with special handling for config types.

        Args:
            config_name: Name of the config attribute (e.g., "model")
            config_class: The config class to instantiate
            matched_kwargs: kwargs that matched the config class fields
            kv_cache_kwargs: kwargs for KVCache config (model config only)
        """
        if config_name == "model" and kv_cache_kwargs:
            # Create new model config with updated KVCache config
            model_config = config_class(**matched_kwargs)

            if self.draft_model:
                memory_util = kv_cache_kwargs.get(
                    "device_memory_utilization", 0.9
                )
                main_model_util = memory_util * 0.7
                draft_model_util = memory_util - main_model_util

                kv_cache_kwargs["device_memory_utilization"] = main_model_util

            model_config.create_kv_cache_config(**kv_cache_kwargs)
            setattr(self, config_name, model_config)

            if self.draft_model:
                kv_cache_kwargs["device_memory_utilization"] = draft_model_util
                self.draft_model.create_kv_cache_config(**kv_cache_kwargs)

        elif config_name == "sampling":
            if hasattr(self, "model") and self.model:
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

            if self.enable_echo or self.draft_model:
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

        model = handler(pydantic_kwargs)
        # `_unmatched_kwargs` is a PrivateAttr, so set it on the instance.
        model._unmatched_kwargs = unmatched_kwargs
        return model

    @model_validator(mode="after")
    def _postprocess_configs(self) -> Self:
        """Process nested configs after Pydantic validation.

        This runs after all fields have been validated and set.
        """
        # Get unmatched kwargs that were stored during preprocessing
        unmatched_kwargs = self._unmatched_kwargs
        if hasattr(self, "_unmatched_kwargs"):
            delattr(self, "_unmatched_kwargs")

        # Process specialized config creation
        self._create_lora_config_if_needed(unmatched_kwargs)
        self._create_draft_model_config_if_needed(unmatched_kwargs)
        self._create_speculative_config_if_needed(unmatched_kwargs)

        # Process remaining config classes
        if unmatched_kwargs:
            self._process_remaining_config_classes(unmatched_kwargs)

        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        # Check both the defer_resolve field and the environment variable
        defer_resolve_env = os.getenv(
            "MODULAR_PIPELINE_DEFER_RESOLVE", ""
        ).lower()
        should_defer = self.defer_resolve or defer_resolve_env in {
            "1",
            "true",
            "yes",
        }
        if not should_defer:
            self.resolve()
        return self

    def retrieve_chat_template(self) -> str | None:
        """Returns the chat template string, or None if not set."""
        # Read the file content
        if self.chat_template is None:
            return None

        try:
            with open(self.chat_template, encoding="utf-8") as f:
                template_content = f.read()

            # Try to parse as JSON and extract chat_template if present
            try:
                template_json = json.loads(template_content)
                if (
                    isinstance(template_json, dict)
                    and "chat_template" in template_json
                ):
                    logger.info(
                        f"Successfully loaded chat_template from JSON in {self.chat_template} "
                        f"({len(template_json['chat_template'])} characters)"
                    )
                    return template_json["chat_template"]
                else:
                    # JSON but no chat_template key, use entire content
                    logger.info(
                        f"Successfully loaded custom prompt template from {self.chat_template} "
                        f"({len(template_content)} characters, JSON without chat_template key)"
                    )
                    return template_content
            except json.JSONDecodeError:
                # Not valid JSON, use entire content as template
                logger.info(
                    f"Successfully loaded custom prompt template from {self.chat_template} "
                    f"({len(template_content)} characters)"
                )
                return template_content

        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(
                f"Failed to read prompt template file {self.chat_template}: {str(e)}. "
                f"Please ensure the file is readable and contains valid UTF-8 text."
            ) from e

    def _resolve_chat_template(self) -> None:
        """Resolves chat_template if it is a Path by reading the file content.

        Handles the case where chat_template is a Path object,
        validates that the file exists, reads its content, and stores the content
        as a string in the chat_template field.

        Raises:
            FileNotFoundError: If the specified template file does not exist
            ValueError: If there's an error reading the template file
        """
        if self.chat_template is None:
            return

        # Expand user home directory if present (e.g., ~/templates/custom.jinja)
        self.chat_template = self.chat_template.expanduser()

        # Convert relative paths to absolute paths
        if not self.chat_template.is_absolute():
            self.chat_template = Path.cwd() / self.chat_template

        # Verify the file exists
        if not self.chat_template.exists():
            raise ValueError(
                f"--chat-template path ({self.chat_template}) does not exist."
            )

        if not self.chat_template.is_file():
            raise ValueError(
                f"Prompt template path is not a file: {self.chat_template}. "
                f"Please provide a path to a valid template file."
            )

    def _import_custom_architectures(self) -> None:
        """Imports custom model modules and adds them to the registry."""
        for module_spec in self.custom_architectures:
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

        # Resolve chat_template if it's a Path
        self._resolve_chat_template()

        self.model.resolve()

        # Validate if a provided max_length is non-negative.
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("max_length must be non-negative.")

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

        # By this point, we should have a valid model_path.

        # Run Baseline Validation
        self._validate_and_resolve_remaining_pipeline_config(
            model_config=self.model
        )

        # Run Additional Checks for Speculative Decoding
        if self.draft_model:
            self._validate_and_resolve_remaining_pipeline_config(
                model_config=self.draft_model
            )

            self._validate_pipeline_config_for_speculative_decoding()

        self._validate_and_resolve_overlap_scheduler()

    def _validate_and_resolve_overlap_scheduler(self) -> None:
        self._validate_and_resolve_device_graph_capture()

        if self.force and not self.device_graph_capture:
            return

        # Automatically enable overlap scheduling for select architectures.
        if not self.enable_overlap_scheduler:
            arch = PIPELINE_REGISTRY.retrieve_architecture(
                huggingface_repo=self.model.huggingface_model_repo,
                use_legacy_module=self.use_legacy_module,
            )
            if (
                arch is not None
                and arch.name
                in (
                    "LlamaForCausalLM_Legacy",
                    "DeepseekV2ForCausalLM_Legacy",
                    "DeepseekV3ForCausalLM_Legacy",
                    "DeepseekV3_2ForCausalLM_Legacy",
                    "DeepseekV3ForCausalLMNextN_Legacy",
                )
                and self.pipeline_role == PipelineRole.PrefillAndDecode
                and not self.sampling.enable_structured_output
                and not self.sampling.enable_variable_logits
                and not self.speculative
                and not self.lora
                and self.model.device_specs[0].device_type != "cpu"
            ):
                self.enable_overlap_scheduler = True
                self.max_num_steps = 1
                logger.info(
                    f"Automatically enabling overlap scheduling for {arch.name} with max-num-steps=1. "
                    "You can manually disable this by setting --no-enable-overlap-scheduler --force."
                )

        # Raise errors when we detect features that are not compatible with the overlap scheduler.
        if self.enable_overlap_scheduler:
            if self.pipeline_role != PipelineRole.PrefillAndDecode:
                raise ValueError(
                    "The Overlap scheduler does not support Disaggregated Inference yet. "
                    "It is only supported with the PrefillAndDecode pipeline role. "
                    f"Found {self.pipeline_role}."
                )
            if self.sampling.enable_structured_output:
                raise ValueError(
                    "Structured outputs are not supported with the Overlap scheduler."
                )
            if self.sampling.enable_variable_logits:
                raise ValueError(
                    "Variable logits are not supported with the Overlap scheduler. "
                )
            if self.speculative:
                raise ValueError(
                    "Speculative decoding is not supported with the Overlap scheduler."
                )
            if self.lora:
                raise ValueError(
                    "LoRA is not supported with the Overlap scheduler."
                )
            if self.max_num_steps > 1:
                raise ValueError(
                    "Max num steps > 1 is not supported with the Overlap scheduler."
                )
            if self.model.device_specs[0].device_type == "cpu":
                raise ValueError(
                    "Overlap scheduler is not supported with CPU models."
                )

    def _validate_and_resolve_device_graph_capture(self) -> None:
        if not self.device_graph_capture:
            return

        # TODO(GENAI-363): Support device graph capture warmup with data
        # parallelism by capturing per-replica inputs.
        if self.model.data_parallel_degree > 1:
            raise ValueError(
                "device_graph_capture currently requires "
                "data_parallel_degree=1."
            )
        if self.max_batch_size is None:
            raise ValueError(
                "device_graph_capture requires max_batch_size to be set."
            )
        if not self.enable_overlap_scheduler:
            logger.info("Enabling overlap scheduling for device graph capture.")
        self.enable_overlap_scheduler = True
        if self.max_num_steps != 1:
            logger.info(
                "Setting max-num-steps=1 for device graph capture with overlap scheduling."
            )
        self.max_num_steps = 1

    def _validate_and_resolve_max_num_steps(self) -> None:
        """Validates and resolves the max_num_steps field (platform-specific)."""
        if self.max_num_steps < 0:
            if self.model.default_device_spec == DeviceSpec.cpu():
                self.max_num_steps = 1
            else:
                self.max_num_steps = 10

    def _validate_pipeline_config_for_speculative_decoding(self) -> None:
        """Validates pipeline config when used in speculative decoding mode."""
        assert self.draft_model is not None
        assert self.speculative is not None

        # Validate that both the `draft_model` and target model `model_path` have the same
        # architecture
        draft_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.draft_model.huggingface_model_repo,
            use_legacy_module=self.use_legacy_module,
        )

        if not draft_arch:
            # Check if a non-legacy version exists when legacy lookup failed
            if self.use_legacy_module:
                non_legacy_arch = PIPELINE_REGISTRY.retrieve_architecture(
                    huggingface_repo=self.draft_model.huggingface_model_repo,
                    use_legacy_module=False,
                )
                if non_legacy_arch:
                    raise ValueError(
                        f"MAX-optimized architecture found for draft model '{self.draft_model.model_path}', "
                        f"but only the new Module-based implementation is available (architecture: '{non_legacy_arch.name}'). "
                        f"Please use the '--no-use-legacy-module' flag to use the new implementation."
                    )
            raise ValueError(
                "MAX-Optimized architecture not found for `draft_model`"
            )

        target_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model.huggingface_model_repo,
            use_legacy_module=self.use_legacy_module,
        )
        if not target_arch:
            # Check if a non-legacy version exists when legacy lookup failed
            if self.use_legacy_module:
                non_legacy_arch = PIPELINE_REGISTRY.retrieve_architecture(
                    huggingface_repo=self.model.huggingface_model_repo,
                    use_legacy_module=False,
                )
                if non_legacy_arch:
                    raise ValueError(
                        f"MAX-optimized architecture found for target model '{self.model.model_path}', "
                        f"but only the new Module-based implementation is available (architecture: '{non_legacy_arch.name}'). "
                        f"Please use the '--no-use-legacy-module' flag to use the new implementation."
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

        if self.enable_echo:
            raise ValueError(
                "enable_echo not currently supported with speculative decoding enabled"
            )

        if self.sampling.enable_structured_output:
            raise ValueError(
                "structured outputs not currently supported with speculative decoding enabled"
            )

        if self.model.kv_cache.enable_prefix_caching and not self.force:
            logging.warning(
                "Prefix caching is not supported with speculative decoding. "
                "Overriding user setting to False. Pass --force to bypass this "
                "validation, though this may result in unexpected behavior or errors."
            )
            self.model.kv_cache.enable_prefix_caching = False
            self.draft_model.kv_cache.enable_prefix_caching = False

    def _validate_and_resolve_remaining_pipeline_config(
        self, model_config: MAXModelConfig
    ) -> None:
        """Updates remaining pipeline config fields if not provided.

        Errors out with a detailed reason if invalid config is provided.
        """
        # Retrieve the architecture
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=model_config.huggingface_model_repo,
            use_legacy_module=self.use_legacy_module,
        )

        # If nothing is provided, we should not update any more params.
        if not arch:
            # Check if a non-legacy version exists when legacy lookup failed
            if self.use_legacy_module:
                non_legacy_arch = PIPELINE_REGISTRY.retrieve_architecture(
                    huggingface_repo=model_config.huggingface_model_repo,
                    use_legacy_module=False,
                )
                if non_legacy_arch:
                    raise ValueError(
                        f"MAX-optimized architecture found for '{model_config.model_path}', "
                        f"but only the new Module-based implementation is available (architecture: '{non_legacy_arch.name}'). "
                        f"Please use the '--no-use-legacy-module' flag to use the new implementation.\n"
                        f"Example: max serve --model-path {model_config.model_path} --no-use-legacy-module"
                    )

            raise ValueError(
                f"MAX-optimized architecture not available for '{model_config.model_path}'. "
                "Please file a request at https://modul.ar/request to add this model architecture to MAX."
            )

        # Validate required arguments
        if not self.force:
            self._validate_required_arguments_against_architecture(arch)

        # Validate that model supports empty batches, if being requested.
        if self.execute_empty_batches and not arch.supports_empty_batches:
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

        # TODO(E2EOPT-28): remove this constraint.
        # Gemma has a MHA head size of 256.
        # This requires a kv cache page size of at least 256.
        if "Gemma3" in arch.name:
            model_config.kv_cache.kv_cache_page_size = max(
                model_config.kv_cache.kv_cache_page_size, 256
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

        if is_diffusion_pipeline(model_config.huggingface_model_repo):
            # Skip memory estimation for diffusion pipelines,
            # since they don't use KV cache.
            return

        # For non-diffusion pipelines, huggingface_config is required
        if model_config.huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}' but could not be loaded. "
                "This model may not be a standard Transformers model. "
                "Please ensure the model repository contains a valid config.json file."
            )

        arch_config = arch.config.initialize(self)

        MemoryEstimator.estimate_memory_footprint(
            self,
            model_config,
            arch_config,
            devices,
            arch.pipeline_model.estimate_weights_size(self),
            arch.pipeline_model.estimate_activation_memory(
                self, model_config.huggingface_config
            ),
        )

        if clamped_max_seq_len := MemoryEstimator.max_supported_sequence_length(
            arch.pipeline_model.estimate_weights_size(self),
            arch.pipeline_model.estimate_activation_memory(
                self, model_config.huggingface_config
            ),
            model_config,
            devices,
            arch_config,
        ):
            if self.max_length is None:
                self.max_length = clamped_max_seq_len
            elif self.max_length > clamped_max_seq_len:
                logging.warning(
                    f"Clamping max_length from {self.max_length} to {clamped_max_seq_len} due to capacity of KV Cache"
                )
                self.max_length = clamped_max_seq_len

        # Validate whether the architecture requires a max batch total tokens to be specified.
        # This needs to be done after max_length is resolved.
        if (
            arch.requires_max_batch_context_length
            and self.max_batch_total_tokens is None
        ):
            logger.warning(
                f"Architecture '{arch.name}' requires max-batch-total-tokens to be specified but found None. "
                f"Defaulting to the max sequence length of the model: {self.max_length}"
            )
            self.max_batch_total_tokens = self.max_length

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
            huggingface_repo=self.model.huggingface_model_repo,
            use_legacy_module=self.use_legacy_module,
        )

        if arch is None:
            raise ValueError(
                f"No architecture found for {self.model.huggingface_model_repo.repo_id}. "
                "This should not happen after config resolution."
            )

        # Get pipeline task and class information
        task = PIPELINE_REGISTRY.retrieve_pipeline_task(self)
        pipeline_class = get_pipeline_for_task(task, self)

        devices_str = ", ".join(
            f"{d.device_type}[{d.id}]" for d in self.model.device_specs
        )

        quantization_encoding_str = str(self.model.quantization_encoding)
        if self.model._applied_dtype_cast_from:
            quantization_encoding_str = f"{quantization_encoding_str} (cast from {self.model._applied_dtype_cast_from})"

        # Build model information entries.
        model_entries: list[tuple[str, Any]] = [
            ("architecture", arch.name),
            ("pipeline_class", pipeline_class.__name__),
            ("pipeline_model", arch.pipeline_model.__name__),
            ("tokenizer", arch.tokenizer_cls.__name__),
            ("devices", devices_str),
            ("model_path", self.model.model_path),
        ]

        # Only show weights_repo_id when it differs from model_path.
        weight_repo_id = self.model.huggingface_weight_repo_id
        if weight_repo_id != self.model.model_path:
            model_entries.append(("weights_repo_id", weight_repo_id))

        model_entries.extend(
            [
                ("huggingface_revision", self.model.huggingface_model_revision),
                ("quantization_encoding", quantization_encoding_str),
            ]
        )

        # Format weight_path depending on the number of paths.
        weight_paths = self.model.weight_path
        if len(weight_paths) == 1:
            model_entries.append(("weight_path", weight_paths[0]))
        else:
            display_paths = (
                weight_paths[:3] + ["..."] + [weight_paths[-1]]
                if len(weight_paths) > 5
                else list(weight_paths)
            )
            formatted = (
                "[\n"
                + "\n".join(f"        {p}" for p in display_paths)
                + "\n    ]"
            )
            model_entries.append(("weight_path", formatted))

        # Log Pipeline and Model Information
        logger.info("")
        logger.info("Model Information")
        logger.info("=" * 60)
        for line in _format_config_entries(model_entries):
            logger.info(line)

        pipeline_entries: list[tuple[str, Any]] = [
            ("max_seq_len", self.max_length),
            ("max_batch_size", self.max_batch_size),
            ("chunked_prefill", self.enable_chunked_prefill),
            ("max_batch_input_tokens", self.max_batch_input_tokens),
            ("in_flight_batching", self.enable_in_flight_batching),
        ]

        logger.info("")
        logger.info("Pipeline Config")
        logger.info("=" * 60)
        for line in _format_config_entries(pipeline_entries):
            logger.info(line)
        logger.info("")

        # KVCache Configuration Summary
        logger.info("KVCache Config")
        logger.info("=" * 60)

        # Primary model kvcache config
        kv_config = self.model.kv_cache
        _log_kvcache_entries(kv_config)

        # Draft model kvcache config (if using speculative decoding)
        if self.draft_model is not None:
            logger.info("")
            logger.info("Draft Model KVCache Configuration:")
            logger.info("-" * 40)
            assert self.draft_model is not None
            draft_kv_config = self.draft_model.kv_cache
            _log_kvcache_entries(draft_kv_config)

        logger.info("")

    def log_basic_config(self) -> None:
        """Log minimal pipeline configuration information.

        Logs basic PipelineConfig options including model name, pipeline task,
        weight path, max_batch_size, max_seq_len, and reserved memory.
        """
        # Retrieve architecture - this should always exist after config resolution
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model.huggingface_model_repo,
            use_legacy_module=self.use_legacy_module,
        )

        if arch is None:
            raise ValueError(
                f"No architecture found for {self.model.huggingface_model_repo.repo_id}. "
                "This should not happen after config resolution."
            )

        task = PIPELINE_REGISTRY.retrieve_pipeline_task(self)
        pipeline_class = get_pipeline_for_task(task, self)

        # Get reserved memory info from KVCache config (only for tasks that use KV cache)
        from max.interfaces.task import PipelineTask

        kv_cache_tasks = {
            PipelineTask.TEXT_GENERATION,
            PipelineTask.AUDIO_GENERATION,
            PipelineTask.SPEECH_TOKEN_GENERATION,
        }

        memory_str = None
        if task in kv_cache_tasks:
            kv_config = self.model.kv_cache
            if kv_config._available_cache_memory is None:
                raise ValueError(
                    "KVCache config is not available after config resolution."
                )
            memory_str = to_human_readable_bytes(
                kv_config._available_cache_memory
            )

        devices_str = ", ".join(
            f"{d.device_type}[{d.id}]" for d in self.model.device_specs
        )

        # Log basic configuration
        config_entries: list[tuple[str, Any]] = (
            [
                ("model", self.model.model_path),
                ("architecture", arch.name),
                ("pipeline", pipeline_class.__name__),
                ("devices", devices_str),
                ("max_batch_size", self.max_batch_size),
                ("max_seq_len", self.max_length),
            ]
            + [("cache_memory", memory_str)]
            if memory_str
            else []
            + [
                ("device_graph_capture", self.device_graph_capture),
            ]
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


def _format_config_entries(
    entries: list[tuple[str, Any]], indent: str = "    "
) -> list[str]:
    """Format key-value config entries with aligned colons.

    Args:
        entries: List of (key, value) tuples to format.
        indent: Prefix string for each line.

    Returns:
        A list of formatted strings with keys left-aligned and colons
        vertically aligned based on the longest key.
    """
    max_key_len = max(len(key) for key, _ in entries)
    return [f"{indent}{key:<{max_key_len}} : {value}" for key, value in entries]


def _log_kvcache_entries(config: KVCacheConfig, indent: str = "    ") -> None:
    """Log KV cache configuration details using aligned formatting.

    Args:
        config: The KVCacheConfig to log.
        indent: Prefix string for each line.
    """
    entries: list[tuple[str, Any]] = [
        ("cache_strategy", config.cache_strategy),
        ("page_size", f"{config.kv_cache_page_size} tokens"),
        ("prefix_caching", config.enable_prefix_caching),
        ("host_swapping", config.enable_kvcache_swapping_to_host),
    ]
    if config.enable_kvcache_swapping_to_host:
        entries.append(
            ("host_swap_space", f"{config.host_kvcache_swap_space_gb} GB")
        )
    entries.append(
        ("memory_utilization", f"{config.device_memory_utilization:.1%}")
    )

    if config._available_cache_memory is None:
        raise ValueError(
            "KVCache config is not available after config resolution."
        )
    entries.append(
        (
            "available_cache_memory",
            to_human_readable_bytes(config._available_cache_memory),
        )
    )

    for line in _format_config_entries(entries, indent=indent):
        logger.info(line)


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


class PrependPromptSpeechTokens(str, Enum):
    NEVER = "never"
    """Never prepend the prompt speech tokens sent to the audio decoder."""

    ONCE = "once"
    """Prepend the prompt speech tokens to the first block of the audio decoder."""

    ROLLING = "rolling"
    """Prepend the prompt speech tokens to the first block of the audio decoder,
    and to later blocks to reach the requested buffer size."""


class PrometheusMetricsMode(str, Enum):
    INSTRUMENT_ONLY = "instrument_only"
    """Instrument metrics through the Prometheus client library, relying on the application to handle the metrics server."""

    LAUNCH_SERVER = "launch_server"
    """Launch a Prometheus server to handle metrics requests."""

    LAUNCH_MULTIPROC_SERVER = "launch_multiproc_server"
    """Launch a Prometheus server in multiprocess mode to report metrics."""


class AudioGenerationConfig(PipelineConfig):
    # TODO: Make these flags more discoverable.
    audio_decoder: str = Field(
        default="",
        description="The name of the audio decoder model architecture.",
    )

    audio_decoder_weights: str = Field(
        default="", description="The path to the audio decoder weights file."
    )

    chunk_size: list[int] | None = Field(
        default=None,
        description=(
            "The chunk sizes to use for streaming. If this is an int, fixed-size "
            "chunks of the given size are used. If this is a list, variable "
            "chunk sizes are used."
        ),
    )

    buffer: int = Field(
        default=0,
        description=(
            "The number of previous speech tokens to pass to the audio decoder "
            "on each generation step."
        ),
    )

    block_causal: bool = Field(
        default=False,
        description=(
            "Whether prior buffered tokens should attend to tokens in the "
            "current block. Has no effect if buffer is not set."
        ),
    )

    prepend_prompt_speech_tokens: PrependPromptSpeechTokens = Field(
        default=PrependPromptSpeechTokens.ONCE,
        description=(
            "Whether the prompt speech tokens should be forwarded to the audio "
            "decoder. Options: never, once, rolling."
        ),
    )

    prepend_prompt_speech_tokens_causal: bool = Field(
        default=False,
        description=(
            "Whether the prompt speech tokens should attend to tokens in the "
            "currently generated audio block. Has no effect if "
            "prepend_prompt_speech_tokens is never."
        ),
    )

    audio_decoder_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the audio decoder model.",
    )

    prometheus_metrics_mode: PrometheusMetricsMode = Field(
        default=PrometheusMetricsMode.INSTRUMENT_ONLY,
        description="The mode to use for Prometheus metrics.",
    )

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
        prepend_prompt_speech_tokens: PrependPromptSpeechTokens = PrependPromptSpeechTokens.NEVER,
        prepend_prompt_speech_tokens_causal: bool = False,
        run_model_test_mode: bool = False,
        prometheus_metrics_mode: PrometheusMetricsMode = PrometheusMetricsMode.INSTRUMENT_ONLY,
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
        """Builds an AudioGenerationConfig from audio CLI flags and config kwargs."""
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

        prepend_prompt_speech_tokens = PrependPromptSpeechTokens(
            audio_flags.pop("prepend_prompt_speech_tokens", "never")
        )

        prepend_prompt_speech_tokens_causal = _parse_flag_bool(
            audio_flags.pop("prepend_prompt_speech_tokens_causal", "false"),
            "prepend_prompt_speech_tokens_causal",
        )

        run_model_test_mode = _parse_flag_bool(
            audio_flags.pop("run_model_test_mode", "false"),
            "run_model_test_mode",
        )

        prometheus_metrics_mode = PrometheusMetricsMode(
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
        if self.force:
            return

        if self.enable_overlap_scheduler:
            raise ValueError(
                "The Overlap scheduler does not support Audio Generation. "
                "Detected AudioGenerationConfig."
            )
