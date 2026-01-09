# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.serve.queue.zmq_queue import generate_zmq_ipc_path
from pydantic import (
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self

from .config_enums import PipelineRole
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

# Default prefill chunk size for chunked prefill and memory estimation.
DEFAULT_PREFILL_CHUNK_SIZE = 8192


class PipelineConfig(ConfigFileModel):
    """Configuration for a pipeline.

    WIP - Once a PipelineConfig is fully initialized, it should be as immutable
    as possible (frozen=True). All underlying dataclass fields should have been
    initialized to their default values, be it user specified via some CLI
    flag, config file, environment variable, or internally set to a reasonable
    default.
    """

    max_length: int | None = Field(default=None)
    """Maximum sequence length of the model."""

    pipeline_role: PipelineRole = Field(default=PipelineRole.PrefillAndDecode)
    """Whether the pipeline should serve both a prefill or decode role or both."""

    max_batch_size: int | None = Field(default=None)
    """Maximum batch size to execute with the model.
    When not specified (None), we determine this value dynamically. For users
    launching in a server scenario, the expectation is that this value should be
    set higher based on server capacity.
    """

    max_ce_batch_size: int = Field(default=192)
    """Maximum cache size to reserve for a single context encoding batch.
    The actual limit is the lesser of this and `max_batch_size`."""

    max_queue_size_tg: int | None = Field(default=None)
    """Maximum number of requests in decode queue. By default, this is max-batch-size."""

    min_batch_size_tg: int | None = Field(default=None)
    """Specifies a soft floor on the decode batch size.

    If the TG batch size is larger than this value, the scheduler will continue to
    run TG batches. If it falls below, the scheduler will prioritize CE. Note that
    this is NOT a strict minimum! By default, this is max-queue-size-tg.

    This is an experimental flag solely for the TTS scheduler. Do not use unless
    you know what you are doing.
    """

    ep_size: int = Field(default=1)
    """The expert parallelism size. Needs to be 1 (no expert parallelism) or the
    total number of GPUs across nodes."""

    ce_delay_ms: float = Field(default=0.0)
    """Duration of scheduler sleep prior to starting a prefill batch.

    This is an experimental flag solely for the TTS scheduler. Do not use unless
    you know what you are doing.
    """

    enable_prioritize_first_decode: bool = Field(default=False)
    """When enabled, the scheduler will always run a TG batch immediately after a CE batch,
    with the same requests. This may be useful for decreasing time-to-first-chunk latency.

    This is an experimental flag solely for the TTS scheduler. Do not use unless
    you know what you are doing.
    """

    experimental_background_queue: bool = Field(default=False)
    """When enabled, offloads queue draining to a background thread for improved performance.

    This is an experimental flag. Use with caution.
    """

    enable_chunked_prefill: bool = Field(default=True)
    """Enable chunked prefill to split context encoding requests into multiple chunks
    based on 'prefill_chunk_size'."""

    enable_in_flight_batching: bool = Field(default=False)
    """When enabled, prioritizes token generation by batching it with context
    encoding requests."""

    max_num_steps: int = Field(default=-1)
    """The number of steps to run for multi-step scheduling. -1 specifies a default value based on
    configuration and platform. Ignored for models which are not auto-regressive (e.g. embedding
    models)."""

    prefill_chunk_size: int = Field(default=DEFAULT_PREFILL_CHUNK_SIZE)
    """The target number of un-encoded tokens to include in each batch.
    This value is used for chunked prefill and memory estimation."""

    enable_echo: bool = Field(default=False)
    """Whether the model should be built with echo capabilities."""

    pool_embeddings: bool = Field(default=True)
    """Whether to pool embedding outputs."""

    chat_template: Path | None = Field(default=None)
    """Optional custom chat template to override the one shipped with the
    HuggingFace model config. Can be either:
    - A Path pointing to a file containing the template

    If a Path is provided, the file will be read during config resolution and
    the content will be stored as a string. This allows customizing the prompt
    formatting for different use cases. If None, the model's default chat
    template will be used."""

    use_experimental_kernels: str = Field(
        default=os.environ.get("USE_EXPERIMENTAL_KERNELS", "false")
    )
    """Enables using experimental mojo kernels with max serve.
    The kernels could be unstable, incorrect, or otherwise have issues.
    """

    use_vendor_blas: str = Field(
        default=os.environ.get("MAX_SERVE_USE_VENDOR_BLAS", "false")
    )
    """Enables using the vendor blas libraries (cublas/hipblas/etc) with max serve.
    Currently, this just replaces matmul calls, but it could replace other numeric functions in the future.
    """

    pdl_level: str = Field(default=os.environ.get("PDL_LEVEL", "0"))
    """Level of overlap of kernel launch via programmatic dependent grid control."""

    custom_architectures: list[str] = Field(default_factory=list)
    """A list of custom architecture implementations to register.
    Each input can either be a raw module name or an import path followed by a colon and the module name.
    Ex:
    - `my_module`
    - `folder/path/to/import:my_module`

    Each module must expose an `ARCHITECTURES` list of architectures to register.
    """

    zmq_endpoint_base: str = Field(default_factory=generate_zmq_ipc_path)
    """The prefix for the ZMQ endpoints used for IPC. This prefix ensures that the
    ZMQ endpoints are unique across multiple MAX Serve instances running on the
    same host. This should be randomly generated when the PipelineConfig is created.
    Ex:
    - lora_request_zmq_endpoint: f"{zmq_endpoint_base}-lora_request"
    - lora_response_zmq_endpoint: f"{zmq_endpoint_base}-lora_response"
    """

    execute_empty_batches: bool = Field(default=False)
    """Whether the scheduler should execute empty batches."""

    max_batch_context_length: int | None = Field(default=None)
    """Ensures that the sum of the context length in a batch does not exceed max_batch_context_length.

    If None, the sum of the context length in batch is not limited.
    """

    force: bool = Field(default=False)
    """Skip validation of user provided flags against the architecture's required arguments."""

    kvcache_ce_watermark: float = Field(default=0.95)
    """Projected cache usage threshold for scheduling CE requests, considers current + incoming
    request. CE is scheduled if either projected usage stays below this threshold OR no active
    requests exist. Greater KVCache utilization (as controlled by this parameter) was
    found to cause more preemptions.
    """

    use_module_v3: bool = Field(default=False)
    """Whether to use the ModuleV3 architecture if it exists."""

    _model: MAXModelConfig = PrivateAttr(default_factory=MAXModelConfig)
    """The model config."""

    _draft_model: MAXModelConfig | None = PrivateAttr(default=None)
    """The draft model config."""

    _sampling: SamplingConfig = PrivateAttr(default_factory=SamplingConfig)
    """The sampling config."""

    _profiling: ProfilingConfig = PrivateAttr(default_factory=ProfilingConfig)
    """The profiling config."""

    _lora: LoRAConfig | None = PrivateAttr(default=None)
    """The LoRA config."""

    _speculative: SpeculativeConfig | None = PrivateAttr(default=None)
    """The SpeculativeConfig."""

    _config_file_section_name: str = PrivateAttr(default="pipeline_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    _unmatched_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    """Temporary storage for unmatched kwargs during initialization.
    This is used to pass unmatched kwargs from the before validator to the after validator."""

    def configure_session(self, session: InferenceSession) -> None:
        """Configure an InferenceSession with standard pipeline settings."""
        session.gpu_profiling(self.profiling_config.gpu_profiling)
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
        """
        Extract kwargs that match a config class's fields.

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
            self._lora = LoRAConfig(**lora_kwargs)
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
            self._draft_model = MAXModelConfig(**draft_kwargs)
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
                self._speculative = SpeculativeConfig(**filtered_kwargs)
                assert self._draft_model is not None
                # We need to set the architecture to EagleLlamaForCausalLM for Eagle speculative decoding
                if self._speculative.is_eagle():
                    assert (
                        len(self._draft_model.huggingface_config.architectures)
                        == 1
                    )
                    hf_arch = (
                        self._draft_model.huggingface_config.architectures[0]
                    )
                    if hf_arch == "LlamaForCausalLM":
                        self._draft_model.huggingface_config.architectures[
                            0
                        ] = "EagleLlamaForCausalLM"

    def _process_remaining_config_classes(
        self, unmatched_kwargs: dict[str, Any]
    ) -> None:
        """
        Process remaining kwargs for other config classes.

        Args:
            unmatched_kwargs: Dictionary of kwargs that haven't been matched yet
        """
        # TODO(zheng): Make this more efficient by using MaxConfig instance
        # instead of hardcoding the config names.
        config_mappings = [
            # NOTE: _model must come before _sampling so that
            # SamplingConfig can use generation_config from the model
            "_model",
            "_sampling",
            "_profiling",
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
                    config_name == "_model"
                    and key in KVCacheConfig.model_fields
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
        """
        Create and set a config object with special handling for different config types.

        Args:
            config_name: Name of the config attribute (e.g., "_model")
            config_class: The config class to instantiate
            matched_kwargs: kwargs that matched the config class fields
            kv_cache_kwargs: kwargs for KVCache config (model config only)
        """
        if config_name == "_model" and kv_cache_kwargs:
            # Create new model config with updated KVCache config
            model_config = config_class(**matched_kwargs)

            if self._draft_model:
                memory_util = kv_cache_kwargs.get(
                    "device_memory_utilization", 0.9
                )
                main_model_util = memory_util * 0.7
                draft_model_util = memory_util - main_model_util

                kv_cache_kwargs["device_memory_utilization"] = main_model_util

            model_config._kv_cache = KVCacheConfig(**kv_cache_kwargs)
            setattr(self, config_name, model_config)

            if self._draft_model:
                kv_cache_kwargs["device_memory_utilization"] = draft_model_util
                self._draft_model._kv_cache = KVCacheConfig(**kv_cache_kwargs)

        elif config_name == "_sampling":
            if hasattr(self, "_model") and self._model:
                assert isinstance(self._model, MAXModelConfig)
                assert hasattr(
                    config_class, "from_generation_config_sampling_defaults"
                )
                sampling_config = config_class.from_generation_config_sampling_defaults(
                    sampling_params_defaults=self._model.sampling_params_defaults,
                    **matched_kwargs,
                )
            else:
                sampling_config = config_class(**matched_kwargs)

            if self.enable_echo or self._draft_model:
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

        self.resolve()
        return self

    def retrieve_chat_template(self) -> str | None:
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
        """
        Resolve chat_template if it's a Path object by reading the file content.

        This method handles the case where chat_template is a Path object,
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
        """
        Import custom model modules to add them to the registry.
        """
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
        """
        Validates and overrides config settings based on required_arguments from SupportedArchitecture.

        This method checks the required_arguments dictionary from the architecture
        and automatically overrides any config values that don't match, logging warnings
        when changes are made.

        Args:
            architecture: The SupportedArchitecture containing required_arguments dictionary
        """
        if not architecture.required_arguments:
            return

        config_objects = [
            ("PipelineConfig", self),
            ("MAXModelConfig", self._model),
            ("SamplingConfig", self._sampling),
            ("KVCacheConfig", self._model._kv_cache),
        ]

        # Add draft model configurations if present
        if self._draft_model is not None:
            config_objects.extend(
                [
                    ("Draft_MAXModelConfig", self._draft_model),
                    (
                        "Draft_KVCacheConfig",
                        self._draft_model._kv_cache,
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
        """
        Validates and resolves the config.

        This method is called after the config is initialized, to ensure that all
        config fields have been initialized to a valid state.
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
            self.sampling_config.enable_structured_output
            and self.model.default_device_spec.device_type == "cpu"
        ):
            raise ValueError(
                "enable_structured_output is not currently supported on CPU."
            )

        if self.sampling_config.enable_penalties and self.draft_model_config:
            logger.warning(
                "frequency_penalty, presence_penalty and repetition_penalty are not currently supported with speculative decoding."
            )
            self.sampling_config.enable_penalties = False

        # Validate LoRA compatibility with model configuration
        if self._lora and self._lora.enable_lora:
            self.model.validate_lora_compatibility()

        # By this point, we should have a valid model_path.

        # Run Baseline Validation
        self._validate_and_resolve_remaining_pipeline_config(
            model_config=self.model
        )

        # Run Additional Checks for Speculative Decoding
        if self.draft_model_config:
            self._validate_and_resolve_remaining_pipeline_config(
                model_config=self.draft_model_config
            )

            self._validate_pipeline_config_for_speculative_decoding()

    def _validate_and_resolve_max_num_steps(self) -> None:
        """
        Validate and resolve the max_num_steps field. These are platform-specific.
        """
        if self.max_num_steps < 0:
            if self.model.default_device_spec == DeviceSpec.cpu():
                self.max_num_steps = 1
            else:
                self.max_num_steps = 10

    def _validate_pipeline_config_for_speculative_decoding(self) -> None:
        """
        Validate the pipeline configs when used in speculative decoding mode.
        """
        assert self.draft_model_config is not None
        assert self._speculative is not None

        # Validate that both the `draft_model` and target model `model_path` have the same
        # architecture
        draft_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.draft_model_config.huggingface_model_repo,
            use_module_v3=self.use_module_v3,
        )

        if not draft_arch:
            raise ValueError(
                "MAX-Optimized architecture not found for `draft_model`"
            )

        target_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model.huggingface_model_repo,
            use_module_v3=self.use_module_v3,
        )
        if not target_arch:
            raise ValueError(
                "MAX-Optimized architecture not found for target model (`model_path`)"
            )

        # Validate that their tokenizers are identical.
        if self._speculative.is_standalone():
            if draft_arch != target_arch:
                raise ValueError(
                    f"architecture for the draft_model ({draft_arch.name}) does not match the architecture retrieved for the target model ({target_arch.name})"
                )

            draft_tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
                huggingface_repo=self.draft_model_config.huggingface_model_repo
            )
            target_tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
                huggingface_repo=self.model.huggingface_model_repo
            )

            # Compare Vocabularies
            if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
                raise ValueError(
                    f"tokenizer for draft_model ({self.draft_model_config.model_path}) does not match the vocabulary of the tokenizer for the target model ({self.model.model_path})"
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
                        f"tokenizer for draft_model ({self.draft_model_config.model_path}) does not match the configuration of the tokenizer for the target model ({self.model.model_path})"
                    )
            else:
                if draft_tokenizer.__dict__ != target_tokenizer.__dict__:
                    raise ValueError(
                        f"tokenizer for draft_model ({self.draft_model_config.model_path}) does not match the configuration of the tokenizer for the target model ({self.model.model_path})"
                    )

        if self.enable_echo:
            raise ValueError(
                "enable_echo not currently supported with speculative decoding enabled"
            )

        if self.sampling_config.enable_structured_output:
            raise ValueError(
                "structured outputs not currently supported with speculative decoding enabled"
            )

        if self.model.kv_cache_config.enable_prefix_caching and not self.force:
            logging.warning(
                "Prefix caching is not supported with speculative decoding. "
                "Overriding user setting to False. Pass --force to bypass this "
                "validation, though this may result in unexpected behavior or errors."
            )
            self.model.kv_cache_config.enable_prefix_caching = False
            self.draft_model_config.kv_cache_config.enable_prefix_caching = (
                False
            )

    def _validate_and_resolve_remaining_pipeline_config(
        self, model_config: MAXModelConfig
    ) -> None:
        """Update remaining pipeline config fields with appropriate values
        if not provided. If invalid config is provided, error out with detailed
        reason."""
        # Retrieve the architecture
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=model_config.huggingface_model_repo,
            use_module_v3=self.use_module_v3,
        )

        # If nothing is provided, we should not update any more params.
        if not arch:
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
        if self._lora and self._lora.enable_lora:
            # Check if the architecture is Llama3 (LlamaForCausalLM)
            if arch.name != "LlamaForCausalLM":
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
            model_config._kv_cache.kv_cache_page_size = max(
                model_config._kv_cache.kv_cache_page_size, 256
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

        # Resolve final pipeline-specific changes to the config before doing
        # memory estimations.
        arch.pipeline_model.finalize_pipeline_config(self)

        MemoryEstimator.estimate_memory_footprint(
            self,
            arch.pipeline_model,
            model_config,
            devices,
        )

        if clamped_max_seq_len := MemoryEstimator.max_supported_sequence_length(
            arch.pipeline_model, self, model_config, devices
        ):
            if self.max_length is None:
                self.max_length = clamped_max_seq_len
            elif self.max_length > clamped_max_seq_len:
                logging.warning(
                    f"Clamping max_length from {self.max_length} to {clamped_max_seq_len} due to capacity of KV Cache"
                )
                self.max_length = clamped_max_seq_len

        # Validate whether the architecture requires a max batch context length to be specified.
        # This needs to be done after max_length is resolved.
        if (
            arch.requires_max_batch_context_length
            and self.max_batch_context_length is None
        ):
            logger.warning(
                f"Architecture '{arch.name}' requires max-batch-context-length to be specified but found None. "
                f"Defaulting to the max sequence length of the model: {self.max_length}"
            )
            self.max_batch_context_length = self.max_length

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
        return self._model.graph_quantization_encoding

    def log_pipeline_info(self) -> None:
        """Log comprehensive pipeline and KVCache configuration information.

        Retrieves all necessary information from self and the PIPELINE_REGISTRY.
        Raises an error if architecture is not found (which should not happen after config resolution).
        """

        # Retrieve architecture - this should always exist after config resolution
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model.huggingface_model_repo,
            use_module_v3=self.use_module_v3,
        )

        if arch is None:
            raise ValueError(
                f"No architecture found for {self.model.huggingface_model_repo.repo_id}. "
                "This should not happen after config resolution."
            )

        # Get pipeline task and class information
        task = PIPELINE_REGISTRY.retrieve_pipeline_task(self)
        pipeline_class = get_pipeline_for_task(task, self)

        # Only show weights_repo_id when it differs from model_path.
        weight_repo_id = self.model.huggingface_weight_repo_id
        weights_repo_str = (
            f"\n            weights_repo_id        : {weight_repo_id}"
            if weight_repo_id != self.model.model_path
            else ""
        )

        devices_str = ", ".join(
            f"{d.device_type}[{d.id}]" for d in self.model.device_specs
        )

        quantization_encoding_str = str(self.model.quantization_encoding)
        if self.model._applied_dtype_cast_from:
            quantization_encoding_str = f"{quantization_encoding_str} (cast from {self.model._applied_dtype_cast_from})"

        # Helper function to log kvcache config details
        def _log_kvcache_details(
            config: KVCacheConfig, indent: str = "    "
        ) -> None:
            logger.info(
                f"{indent}cache_strategy         : {config.cache_strategy}"
            )
            logger.info(
                f"{indent}page_size              : {config.kv_cache_page_size} tokens"
            )
            logger.info(
                f"{indent}prefix_caching         : {config.enable_prefix_caching}"
            )
            logger.info(
                f"{indent}host_swapping          : {config.enable_kvcache_swapping_to_host}"
            )
            if config.enable_kvcache_swapping_to_host:
                logger.info(
                    f"{indent}host_swap_space        : {config.host_kvcache_swap_space_gb} GB"
                )
            logger.info(
                f"{indent}memory_utilization     : {config.device_memory_utilization:.1%}"
            )

            if config._available_cache_memory is None:
                raise ValueError(
                    "KVCache config is not available after config resolution."
                )
            logger.info(
                f"{indent}available_cache_memory : {to_human_readable_bytes(config._available_cache_memory)}"
            )

        # Log Pipeline and Model Information
        logger.info("")
        logger.info("Model Information")
        logger.info("=" * 60)
        logger.info(f"    architecture           : {arch.name}")
        logger.info(f"    pipeline_class         : {pipeline_class.__name__}")
        logger.info(
            f"    pipeline_model         : {arch.pipeline_model.__name__}"
        )
        logger.info(
            f"    tokenizer              : {arch.tokenizer_cls.__name__}"
        )
        logger.info(f"    devices                : {devices_str}")
        logger.info(
            f"    model_path             : {self.model.model_path}{weights_repo_str}"
        )
        logger.info(
            f"    huggingface_revision   : {self.model.huggingface_model_revision}"
        )
        logger.info(f"    quantization_encoding  : {quantization_encoding_str}")

        if len(self.model.weight_path) == 1:
            # Single weight path - format inline
            logger.info(
                f"    weight_path            : {self.model.weight_path[0]}"
            )
        elif len(self.model.weight_path) > 5:
            # Many weight paths - replace middle with "..."
            logger.info("    weight_path            : [")
            for path in (
                self.model.weight_path[:3]
                + ["..."]
                + [self.model.weight_path[-1]]
            ):
                logger.info(f"                              {path}")
            logger.info("                            ]")
        else:
            # Few weight paths - print all of them
            logger.info("    weight_path            : [")
            for path in self.model.weight_path:
                logger.info(f"                              {path}")
            logger.info("                            ]")

        logger.info("")
        logger.info("Pipeline Config")
        logger.info("=" * 60)
        logger.info(f"    max_seq_len            : {self.max_length}")
        logger.info(f"    max_batch_size         : {self.max_batch_size}")
        logger.info(
            f"    chunked_prefill        : {self.enable_chunked_prefill}"
        )
        logger.info(f"    prefill_chunk_size     : {self.prefill_chunk_size}")
        logger.info(
            f"    in_flight_batching     : {self.enable_in_flight_batching}"
        )
        logger.info("")

        # KVCache Configuration Summary
        logger.info("KVCache Config")
        logger.info("=" * 60)

        # Primary model kvcache config
        kv_config = self.model._kv_cache
        _log_kvcache_details(kv_config)

        # Draft model kvcache config (if using speculative decoding)
        if self.draft_model_config is not None:
            logger.info("")
            logger.info("Draft Model KVCache Configuration:")
            logger.info("-" * 40)
            assert self._draft_model is not None
            draft_kv_config = self._draft_model._kv_cache
            _log_kvcache_details(draft_kv_config)

        logger.info("")

    def log_basic_config(self) -> None:
        """Log minimal pipeline configuration information.

        Logs basic PipelineConfig options including model name, pipeline task,
        weight path, max_batch_size, max_seq_len, and reserved memory.
        """
        # Retrieve architecture - this should always exist after config resolution
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model.huggingface_model_repo,
            use_module_v3=self.use_module_v3,
        )

        if arch is None:
            raise ValueError(
                f"No architecture found for {self.model.huggingface_model_repo.repo_id}. "
                "This should not happen after config resolution."
            )

        # Get pipeline task
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model.huggingface_model_repo,
            use_module_v3=self.use_module_v3,
        )
        if arch is None:
            raise ValueError(
                f"No architecture found for {self.model.huggingface_model_repo.repo_id}. "
                "This should not happen after config resolution."
            )

        task = PIPELINE_REGISTRY.retrieve_pipeline_task(self)
        pipeline_class = get_pipeline_for_task(task, self)

        # Get reserved memory info from KVCache config
        kv_config = self.model._kv_cache
        if kv_config._available_cache_memory is None:
            raise ValueError(
                "KVCache config is not available after config resolution."
            )
        memory_str = to_human_readable_bytes(kv_config._available_cache_memory)

        devices_str = ", ".join(
            f"{d.device_type}[{d.id}]" for d in self.model.device_specs
        )

        # Log basic configuration
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "Pipeline Configuration (use --pretty-print-config to print full config)"
        )
        logger.info("=" * 60)
        logger.info(f"    model              : {self.model.model_path}")
        logger.info(f"    architecture       : {arch.name}")
        logger.info(f"    pipeline           : {pipeline_class.__name__}")
        logger.info(f"    devices            : {devices_str}")
        logger.info(f"    max_batch_size     : {self.max_batch_size}")
        logger.info(f"    max_seq_len        : {self.max_length}")
        logger.info(f"    cache_memory       : {memory_str}")
        logger.info("")

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "max_length": "Set the maximum sequence length for input data processed by the model. This must be less than the value specified in the Hugging Face configuration file. The default is derived from the Hugging Face configuration value. Larger values may consume more memory.",
            "pipeline_role": "Whether the pipeline should serve both a prefill or decode role or both.",
            "max_batch_size": "Define the maximum batch size to execute with the model. When not specified (None), we determine this value dynamically. For users launching in a server scenario, the expectation is that this value should be set higher based on server capacity.",
            "max_queue_size_tg": "Maximum number of requests in decode queue. By default, this is max-batch-size.",
            "min_batch_size_tg": "Specifies a soft floor on the decode batch size. If the TG batch size is larger than this value, the scheduler will continue to run TG batches. If it falls below, the scheduler will prioritize CE. This is an experimental flag solely for the TTS scheduler.",
            "ce_delay_ms": "Duration of scheduler sleep prior to starting a prefill batch. This is an experimental flag solely for the TTS scheduler. Default is 0.0.",
            "enable_prioritize_first_decode": "When enabled, the scheduler will always run a TG batch immediately after a CE batch, with the same requests. This may be useful for decreasing time-to-first-chunk latency. This is an experimental flag solely for the TTS scheduler. Default is false.",
            "experimental_background_queue": "When enabled, offloads queue draining to a background thread for improved performance. This is an experimental flag. Default is false.",
            "enable_chunked_prefill": "Enable chunked prefill to split context encoding requests into multiple chunks based on `prefill-chunk-size`. Default is true.",
            "enable_in_flight_batching": "When enabled, prioritizes token generation by batching it with context encoding requests. Default is false.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is -1 which specifies a default value based on configuration and platform. Ignored for models which are not auto-regressive (e.g. embedding models).",
            "prefill_chunk_size": "The target number of un-encoded tokens to include in each batch. This value is used for chunked prefill and memory estimation. Default is 8192.",
            "enable_echo": "Whether the model should be built with echo capabilities. This defaults to false.",
            "pool_embeddings": "Whether to pool embedding outputs. Default is true.",
            "use_experimental_kernels": "Whether to use experimental kernels. Default is false.",
            "max_batch_context_length": "Ensures that the sum of the context length in a batch does not exceed max_batch_context_length. If None, the sum of the context length in batch is not limited.",
            "pdl_level": "Level of overlap of kernel launch via programmatic dependent grid control. Default is 0.",
            "custom_architectures": "A list of custom architecture implementations to register. Each input can either be a raw module name or an import path followed by a colon and the module name.",
            "kvcache_ce_watermark": "Projected cache usage threshold for scheduling CE requests, considers current + incoming request. CE is scheduled if either projected usage stays below this threshold OR no active requests exist. Greater KVCache utilization (as controlled by this parameter) was found to cause more preemptions. Default watermark value is 0.95.",
        }

    @property
    def model(self) -> MAXModelConfig:
        """Accessor for the MAXModelConfig instance."""
        return self._model

    # TODO(SERVSYS-1066): This should be renamed to just draft_model.
    @property
    def draft_model_config(self) -> MAXModelConfig | None:
        return self._draft_model

    # TODO(SERVSYS-1066): This should be renamed to just sampling.
    @property
    def sampling_config(self) -> SamplingConfig:
        return self._sampling

    # TODO(SERVSYS-1066): This should be renamed to just profiling.
    @property
    def profiling_config(self) -> ProfilingConfig:
        return self._profiling

    # TODO(SERVSYS-1066): This should be renamed to just lora.
    @property
    def lora_config(self) -> LoRAConfig | None:
        return self._lora


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
    audio_decoder: str = Field(default="")
    """The name of the audio decoder model architecture."""

    audio_decoder_weights: str = Field(default="")
    """The path to the audio decoder weights file."""

    chunk_size: list[int] | None = Field(default=None)
    """The chunk sizes to use for streaming.
    If this is an int, then fixed-size chunks of the given size are used
    If this is a list, then variable chunk sizes are used."""

    buffer: int = Field(default=0)
    """The number of previous speech tokens to pass to the audio decoder on
    each generation step."""

    block_causal: bool = Field(default=False)
    """Whether prior buffered tokens should attend to tokens in the current block.
    Has no effect if buffer is not set."""

    prepend_prompt_speech_tokens: PrependPromptSpeechTokens = Field(
        default=PrependPromptSpeechTokens.ONCE
    )
    """Whether the prompt speech tokens should be forwarded to the audio decoder.
    If "never", the prompt tokens are not forwarded.
    If "once", the prompt tokens are only forwarded on the first block.
    If "always", the prompt tokens are forwarded on all blocks.
    """

    prepend_prompt_speech_tokens_causal: bool = Field(default=False)
    """Whether the prompt speech tokens should attend to tokens in the currently
    generated audio block.
    Has no effect if prepend_prompt_speech_tokens is "never".
    If False (default), the prompt tokens do not attend to the current block.
    If True, the prompt tokens attend to the current block.
    """

    audio_decoder_config: dict[str, Any] = Field(default_factory=dict)
    """Parameters to pass to the audio decoder model."""

    prometheus_metrics_mode: PrometheusMetricsMode = Field(
        default=PrometheusMetricsMode.INSTRUMENT_ONLY
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

    @staticmethod
    def help() -> dict[str, str]:
        # Get the parent class help first
        audio_help = PipelineConfig.help()

        # Add AudioGenerationConfig-specific fields
        audio_specific_help = {
            "audio_decoder": "The name of the audio decoder model architecture.",
            "audio_decoder_weights": "The path to the audio decoder weights file.",
            "chunk_size": "The chunk sizes to use for streaming. If this is an int, then fixed-size chunks of the given size are used. If this is a list, then variable chunk sizes are used.",
            "buffer": "The number of previous speech tokens to pass to the audio decoder on each generation step. Default is 0.",
            "block_causal": "Whether prior buffered tokens should attend to tokens in the current block. Has no effect if buffer is not set. Default is false.",
            "prepend_prompt_speech_tokens": "Whether the prompt speech tokens should be forwarded to the audio decoder. Options: 'never', 'once', 'rolling'. Default is 'once'.",
            "prepend_prompt_speech_tokens_causal": "Whether the prompt speech tokens should attend to tokens in the currently generated audio block. Has no effect if prepend_prompt_speech_tokens is 'never'. Default is false.",
            "audio_decoder_config": "Parameters to pass to the audio decoder model.",
            "prometheus_metrics_mode": "The mode to use for Prometheus metrics. Default is 'instrument_only'.",
        }

        # Check for conflicts
        for key in audio_specific_help:
            if key in audio_help:
                raise ValueError(
                    f"Duplicate help key '{key}' found in AudioGenerationConfig"
                )

        # Merge the help dictionaries
        audio_help.update(audio_specific_help)
        return audio_help

    @classmethod
    def from_flags(
        cls, audio_flags: dict[str, str], **config_flags: Any
    ) -> AudioGenerationConfig:
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
