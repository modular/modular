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
import logging
import os
import sys
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from typing import Any, Optional, get_type_hints

from max.driver import DeviceSpec, load_devices
from max.graph.quantization import QuantizationEncoding

from .config_enums import PipelineRole
from .kv_cache_config import KVCacheConfig
from .lora_config import LoRAConfig
from .max_config import MAXConfig
from .memory_estimation import MEMORY_ESTIMATOR
from .model_config import MAXModelConfig
from .profiling_config import ProfilingConfig
from .registry import PIPELINE_REGISTRY, get_pipeline_for_task
from .sampling import SamplingConfig

logger = logging.getLogger("max.pipelines")

# Default prefill chunk size for chunked prefill and memory estimation.
DEFAULT_PREFILL_CHUNK_SIZE = 8192


@dataclass(frozen=False)
class PipelineConfig(MAXConfig):
    """Configuration for a pipeline.

    WIP - Once a PipelineConfig is fully initialized, it should be as immutable
    as possible (frozen=True). All underlying dataclass fields should have been
    initialized to their default values, be it user specified via some CLI
    flag, config file, environment variable, or internally set to a reasonable
    default.
    """

    max_length: Optional[int] = None
    """Maximum sequence length of the model."""

    pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode
    """Whether the pipeline should serve both a prefill or decode role or both."""

    max_batch_size: Optional[int] = None
    """Maximum batch size to execute with the model.
    When not specified (None), we determine this value dynamically. For users
    launching in a server scenario, the expectation is that this value should be
    set higher based on server capacity.
    """

    max_ce_batch_size: int = 192
    """Maximum cache size to reserve for a single context encoding batch.
    The actual limit is the lesser of this and `max_batch_size`."""

    max_queue_size_tg: Optional[int] = None
    """Maximum number of requests in decode queue. By default, this is max-batch-size."""

    min_batch_size_tg: Optional[int] = None
    """Specifies a soft floor on the decode batch size.

    If the TG batch size is larger than this value, the scheduler will continue to
    run TG batches. If it falls below, the scheduler will prioritize CE. Note that
    this is NOT a strict minimum! By default, this is max-queue-size-tg.

    This is an experimental flag solely for the TTS scheduler. Do not use unless
    you know what you are doing.
    """

    ce_delay_ms: float = 0.0
    """Duration of scheduler sleep prior to starting a prefill batch.

    This is an experimental flag solely for the TTS scheduler. Do not use unless
    you know what you are doing.
    """

    enable_prioritize_first_decode: bool = False
    """When enabled, the scheduler will always run a TG batch immediately after a CE batch,
    with the same requests. This may be useful for decreasing time-to-first-chunk latency.

    This is an experimental flag solely for the TTS scheduler. Do not use unless
    you know what you are doing.
    """

    enable_chunked_prefill: bool = True
    """Enable chunked prefill to split context encoding requests into multiple chunks
    based on 'prefill_chunk_size'."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context
    encoding requests."""

    max_num_steps: int = -1
    """The number of steps to run for multi-step scheduling. -1 specifies a default value based on
    configuration and platform. Ignored for models which are not auto-regressive (e.g. embedding
    models)."""

    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE
    """The target number of un-encoded tokens to include in each batch.
    This value is used for chunked prefill and memory estimation."""

    enable_echo: bool = False
    """Whether the model should be built with echo capabilities."""

    pool_embeddings: bool = True
    """Whether to pool embedding outputs."""

    use_experimental_kernels: str = os.environ.get(
        "USE_EXPERIMENTAL_KERNELS", "false"
    )

    pdl_level: str = os.environ.get("PDL_LEVEL", "0")
    """Level of overlap of kernel launch via programmatic dependent grid control."""

    custom_architectures: list[str] = field(default_factory=list)
    """A list of custom architecture implementations to register.
    Each input can either be a raw module name or an import path followed by a colon and the module name.
    Ex:
    - `my_module`
    - `folder/path/to/import:my_module`

    Each module must expose an `ARCHITECTURES` list of architectures to register.
    """

    _model_config: MAXModelConfig = field(default_factory=MAXModelConfig)
    """The model config."""

    _draft_model_config: Optional[MAXModelConfig] = None
    """The draft model config."""

    _sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    """The sampling config."""

    _profiling_config: ProfilingConfig = field(default_factory=ProfilingConfig)
    """The profiling config."""

    _lora_config: Optional[LoRAConfig] = None
    """The LoRA config."""

    _config_file_section_name: str = "pipeline_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @staticmethod
    def _extract_kwargs_for_config(
        kwargs: dict[str, Any],
        config_class: type[MAXConfig],
        key_prefix: str = "",
        strip_prefix: bool = False,
    ) -> dict[str, Any]:
        """
        Extract kwargs that match a config class's fields.

        Args:
            kwargs: Source kwargs dictionary (modified in place)
            config_class: The MAXConfig dataclass to match fields against
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

            # Check if this field exists in the config class
            if field_name in config_class.__dataclass_fields__:
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
            self._lora_config = LoRAConfig(**lora_kwargs)
        # TODO: We should add an elif to check / error out if other LoRA params
        # are provided, but enable_lora is not. We can't do this today as our
        # click PipelineConfig autogenerates defaults for all fields, including
        # required ones.

    def _create_draft_model_config_if_needed(
        self, kwargs: dict[str, Any]
    ) -> None:
        """Extract draft model kwargs and create MAXModelConfig if model_path provided."""
        draft_kwargs = PipelineConfig._extract_kwargs_for_config(
            kwargs, MAXModelConfig, key_prefix="draft_", strip_prefix=True
        )

        if draft_kwargs.get("model_path", "") != "":
            self._draft_model_config = MAXModelConfig(**draft_kwargs)
        # TODO: We should add an elif to check / error out if other draft model
        # params are provided, but model_path is not. We can't do this today
        # as our click PipelineConfig autogenerates defaults for all fields,
        # including required ones.

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
            "_sampling_config",
            "_profiling_config",
            # TODO(zheng): Remove this once backward compatibility is no
            # longer needed for MAXModelConfig.
            "_model_config",
        ]

        for config_name in config_mappings:
            config_class = get_type_hints(self.__class__)[config_name]
            matched_kwargs = {}
            kv_cache_kwargs = {}

            for key, value in unmatched_kwargs.items():
                if key in config_class.__dataclass_fields__:
                    matched_kwargs[key] = value
                # Check if this is a KVCache config param
                elif (
                    config_name == "_model_config"
                    and key in KVCacheConfig.__dataclass_fields__
                ):
                    kv_cache_kwargs[key] = value

            if matched_kwargs:
                self._create_and_set_config(
                    config_name, config_class, matched_kwargs, kv_cache_kwargs
                )

                # Remove matched kwargs
                for key in matched_kwargs:
                    del unmatched_kwargs[key]
                for key in kv_cache_kwargs:
                    del unmatched_kwargs[key]

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
            config_name: Name of the config attribute (e.g., "_model_config")
            config_class: The config class to instantiate
            matched_kwargs: kwargs that matched the config class fields
            kv_cache_kwargs: kwargs for KVCache config (model config only)
        """
        if config_name == "_model_config" and kv_cache_kwargs:
            # Create new model config with updated KVCache config
            model_config = config_class(**matched_kwargs)
            model_config._kv_cache_config = KVCacheConfig(**kv_cache_kwargs)
            setattr(self, config_name, model_config)

            if self._draft_model_config:
                self._draft_model_config._kv_cache_config = KVCacheConfig(
                    **kv_cache_kwargs
                )

        elif config_name == "_sampling_config" and (
            self.enable_echo or self._draft_model_config
        ):
            sampling_config = config_class(**matched_kwargs)
            sampling_config.enable_variable_logits = True
            setattr(self, config_name, sampling_config)
        else:
            setattr(self, config_name, config_class(**matched_kwargs))

    def __init__(self, **kwargs: Any) -> None:
        # Initialize all fields with their defaults first
        for curr_field in fields(self.__class__):
            if curr_field.default is not MISSING:
                setattr(self, curr_field.name, curr_field.default)
            elif curr_field.default_factory is not MISSING:
                setattr(self, curr_field.name, curr_field.default_factory())

        # Process specialized config creation
        self._create_lora_config_if_needed(kwargs)
        self._create_draft_model_config_if_needed(kwargs)

        # Check if any kwargs are meant for other MAXConfig classes
        unmatched_kwargs: dict[str, Any] = {}
        # Then process kwargs which override defaults
        for key, value in list(kwargs.items()):
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
                del kwargs[key]
            else:
                unmatched_kwargs[key] = value

        # Process remaining config classes
        if unmatched_kwargs:
            self._process_remaining_config_classes(unmatched_kwargs)

        # NOTE: Do not use this directly after instantiating PipelineConfig. We
        # only keep this here to support backward compatibility of the draft_model
        # field entrypoint. This will be removed entirely soon. I purposefully
        # set this to an empty string than None, to ensure that we catch any
        # inadvertent use of draft_model.
        self.draft_model = ""
        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        self.resolve()

    def _import_custom_architectures(self) -> None:
        """
        Import custom model modules to add them to the registry.
        """
        for module_spec in self.custom_architectures:
            module_parts = module_spec.split(":")
            if len(module_parts) > 2:
                msg = f"Custom module spec contains too many colons: {module_spec}"
                raise ValueError(msg)
            elif len(module_parts) == 2:
                module_path, module_name = module_parts
            else:
                module_path = os.path.dirname(module_parts[0])
                module_name = os.path.basename(module_parts[0])
            sys.path.append(module_path)
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                msg = f"Failed to import custom model from: {module_spec}"
                raise ValueError(msg) from e

            if not module.ARCHITECTURES or not isinstance(
                module.ARCHITECTURES, list
            ):
                msg = f"Custom model imported, but did not expose an `ARCHITECTURES` list. Module: {module_spec}"
                raise ValueError(msg)

            for arch in module.ARCHITECTURES:
                PIPELINE_REGISTRY.register(arch, allow_override=True)

    def resolve(self) -> None:
        """
        Validates and resolves the config.

        This method is called after the config is initialized, to ensure that all
        config fields have been initialized to a valid state.
        """
        # Before anything else, import custom model modules to add them to the registry.
        self._import_custom_architectures()

        self.model_config.resolve()
        # Validate if a provided max_length is non-negative.
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("max_length must be non-negative.")

        self._validate_and_resolve_max_num_steps()

        if (
            self.sampling_config.enable_structured_output
            and self.model_config.default_device_spec.device_type == "cpu"
        ):
            raise ValueError(
                "enable_structured_output is not currently supported on CPU."
            )

        if self.sampling_config.do_penalties and self.draft_model_config:
            raise ValueError(
                "frequency_penalty, presence_penalty and repetition_penalty are not currently supported with speculative decoding."
            )

        # By this point, we should have a valid model_path.

        # Run Baseline Validation
        self._validate_and_resolve_remaining_pipeline_config(
            model_config=self.model_config
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
            if (
                self.sampling_config.enable_structured_output
                or self.model_config.default_device_spec == DeviceSpec.cpu()
            ):
                self.max_num_steps = 1
            else:
                self.max_num_steps = 10

        if (
            self.max_num_steps > 1
            and self.sampling_config.enable_structured_output
        ):
            raise ValueError(
                "max_num_steps > 1 not supported when enable_structured_output = True"
            )

    def _validate_pipeline_config_for_speculative_decoding(self) -> None:
        """
        Validate the pipeline configs when used in speculative decoding mode.
        """
        assert self.draft_model_config is not None  # keep mypy happy

        # Validate that both the `draft_model` and target model `model_path` have the same
        # architecture
        draft_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.draft_model_config.huggingface_model_repo
        )

        if not draft_arch:
            msg = "MAX-Optimized architecture not found for `draft_model`"
            raise ValueError(msg)

        target_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model_config.huggingface_model_repo
        )
        if not target_arch:
            msg = "MAX-Optimized architecture not found for target model (`model_path`)"
            raise ValueError(msg)

        if draft_arch != target_arch:
            msg = f"architecture for the draft_model ({draft_arch.name}) does not match the architecture retrieved for the target model ({target_arch.name})"
            raise ValueError(msg)

        # Validate that their tokenizers are identical.
        draft_tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
            huggingface_repo=self.draft_model_config.huggingface_model_repo
        )
        target_tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
            huggingface_repo=self.model_config.huggingface_model_repo
        )

        # Compare Vocabularies
        if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
            msg = f"tokenizer for draft_model ({self.draft_model_config.model_path}) does not match the vocabulary of the tokenizer for the target model ({self.model_config.model_path})"
            raise ValueError(msg)

        # Compare Tokenizer Configuration
        if hasattr(draft_tokenizer, "_tokenizer") and hasattr(
            target_tokenizer, "_tokenizer"
        ):
            if (
                draft_tokenizer._tokenizer.__dict__
                != target_tokenizer._tokenizer.__dict__
            ):
                msg = f"tokenizer for draft_model ({self.draft_model_config.model_path}) does not match the configuration of the tokenizer for the target model ({self.model_config.model_path})"
                raise ValueError(msg)
        else:
            if draft_tokenizer.__dict__ != target_tokenizer.__dict__:
                msg = f"tokenizer for draft_model ({self.draft_model_config.model_path}) does not match the configuration of the tokenizer for the target model ({self.model_config.model_path})"
                raise ValueError(msg)

        if self.enable_echo:
            msg = "enable_echo not currently supported with speculative decoding enabled"
            raise ValueError(msg)

        if self.sampling_config.enable_structured_output:
            msg = "structured outputs not currently supported with speculative decoding enabled"
            raise ValueError(msg)

    def _validate_and_resolve_remaining_pipeline_config(
        self, model_config: MAXModelConfig
    ) -> None:
        """Update remaining pipeline config fields with appropriate values
        if not provided. If invalid config is provided, error out with detailed
        reason."""
        # Retrieve the architecture
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=model_config.huggingface_model_repo
        )

        # If nothing is provided, we should not update any more params.
        if not arch:
            msg = (
                f"MAX-optimized architecture not available for '{model_config.model_path}'. "
                "Please file a request at https://modul.ar/request to add this model architecture to MAX."
            )
            raise ValueError(msg)

        model_config.validate_prefix_caching_supported(
            prefix_caching_supported=arch.prefix_caching_supported
        )

        # TODO(E2EOPT-28): remove this constraint.
        # Gemma has a MHA head size of 256.
        # This requires a kv cache page size of at least 256.
        if "Gemma3" in arch.name:
            model_config._kv_cache_config.kv_cache_page_size = max(
                model_config._kv_cache_config.kv_cache_page_size, 256
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
            msg = f"quantization_encoding of '{model_config.quantization_encoding}' not supported by MAX engine."
            raise ValueError(msg)

        model_config.validate_and_resolve_with_resolved_quantization_encoding(
            supported_encodings=arch.supported_encodings,
            default_weights_format=arch.default_weights_format,
        )

        devices = load_devices(model_config.device_specs)
        MEMORY_ESTIMATOR.estimate_memory_footprint(
            self, arch.pipeline_model, model_config, devices
        )

    def __getstate__(self) -> dict[str, Any]:
        """Override `__getstate__` to exclude the Hugging Face config."""
        state = self.__dict__.copy()
        return state

    @property
    def graph_quantization_encoding(self) -> Optional[QuantizationEncoding]:
        """Converts the CLI encoding to a MAX graph quantization encoding.

        Returns:
            The graph quantization encoding corresponding to the CLI encoding.
        """
        return self._model_config.graph_quantization_encoding

    def log_pipeline_info(self) -> None:
        """Log comprehensive pipeline and KVCache configuration information.

        Retrieves all necessary information from self and the PIPELINE_REGISTRY.
        Raises an error if architecture is not found (which should not happen after config resolution).
        """

        # Retrieve architecture - this should always exist after config resolution
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model_config.huggingface_model_repo
        )

        if arch is None:
            raise ValueError(
                f"No architecture found for {self.model_config.huggingface_model_repo.repo_id}. "
                "This should not happen after config resolution."
            )

        # Get pipeline task and class information
        task = PIPELINE_REGISTRY.retrieve_pipeline_task(self)
        pipeline_class = get_pipeline_for_task(task, self)
        devices = load_devices(self.model_config.device_specs)

        # Prepare logging information
        if len(self.model_config.weight_path) == 1:
            # Single weight path - keep it inline
            weight_path = f" {self.model_config.weight_path[0]} "
        else:
            # Multiple weight paths - format each on a new line with proper indentation
            weight_path = ",\n                                ".join(
                f"{path}" for path in self.model_config.weight_path
            )

        weights_repo_str = (
            f"\n            weights_repo_id:        {self.model_config._weights_repo_id}"
            if self.model_config._weights_repo_id
            else ""
        )

        devices_str = ", ".join(f"{d.label}[{d.id}]" for d in devices)

        quantization_encoding_str = str(self.model_config.quantization_encoding)
        if self.model_config._applied_dtype_cast_from:
            quantization_encoding_str = f"{quantization_encoding_str} (cast from {self.model_config._applied_dtype_cast_from})"

        # Helper function to log kvcache config details
        def _log_kvcache_details(config: KVCacheConfig, indent: str = "    "):
            logger.info(
                f"{indent}cache_strategy:         {config.cache_strategy}"
            )
            logger.info(
                f"{indent}page_size:              {config.kv_cache_page_size} tokens"
            )
            logger.info(
                f"{indent}prefix_caching:         {config.enable_prefix_caching}"
            )
            logger.info(
                f"{indent}host_swapping:          {config.enable_kvcache_swapping_to_host}"
            )
            logger.info(
                f"{indent}memory_utilization:     {config.device_memory_utilization:.1%}"
            )
            logger.info(
                f"{indent}host_swap_space:        {config.host_kvcache_swap_space_gb} GB"
            )

            if config._available_cache_memory is not None:
                cache_gb = config._available_cache_memory / (1024**3)
                logger.info(
                    f"{indent}available_cache_memory: {cache_gb:.2f} GB ({config._available_cache_memory} bytes)"
                )
            else:
                logger.info(
                    f"{indent}available_cache_memory: Not calculated yet"
                )

        # Log Pipeline and Model Information
        logger.info("")
        logger.info("Model Information")
        logger.info("=" * 60)
        logger.info(f"    architecture:           {arch.name}")
        logger.info(f"    task:                   {task}")
        logger.info(f"    pipeline_class:         {pipeline_class.__name__}")
        logger.info(
            f"    pipeline_model:         {arch.pipeline_model.__name__}"
        )
        logger.info(
            f"    tokenizer:              {arch.tokenizer_cls.__name__}"
        )
        logger.info(f"    devices:                {devices_str}")
        logger.info(
            f"    model_path:             {self.model_config.model_path}{weights_repo_str}"
        )
        logger.info(
            f"    huggingface_revision:   {self.model_config.huggingface_model_revision}"
        )
        logger.info(f"    quantization_encoding:  {quantization_encoding_str}")
        if len(self.model_config.weight_path) == 1:
            # Single weight path - format inline
            logger.info(f"    weight_path:            [{weight_path}]")
        else:
            # Multiple weight paths - format on multiple lines
            logger.info("    weight_path:            [")
            logger.info(f"                                {weight_path}")
            logger.info("                                ]")
        logger.info("")
        logger.info("Pipeline Runtime Configuration:")
        logger.info("-" * 40)
        logger.info(f"    max_seq_len:    {self.max_length}")
        logger.info(f"    max_batch_size:         {self.max_batch_size}")
        logger.info(f"    max_ce_batch_size:      {self.max_ce_batch_size}")
        logger.info(
            f"    chunked_prefill:        {self.enable_chunked_prefill}"
        )
        logger.info(f"    prefill_chunk_size:     {self.prefill_chunk_size}")
        logger.info(
            f"    in_flight_batching:     {self.enable_in_flight_batching}"
        )
        logger.info("")

        # KVCache Configuration Summary
        logger.info("KVCache Config")
        logger.info("=" * 60)

        # Primary model kvcache config
        kv_config = self.model_config._kv_cache_config
        _log_kvcache_details(kv_config)

        # Draft model kvcache config (if using speculative decoding)
        if self.draft_model_config is not None:
            logger.info("")
            logger.info("Draft Model KVCache Configuration:")
            logger.info("-" * 40)
            draft_kv_config = self.draft_model_config._kv_cache_config
            _log_kvcache_details(draft_kv_config)

        logger.info("")

    def log_basic_config(self) -> None:
        """Log minimal pipeline configuration information.

        Logs basic PipelineConfig options including model name, pipeline task,
        weight path, max_batch_size, max_seq_len, and reserved memory.
        """
        # Retrieve architecture - this should always exist after config resolution
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model_config.huggingface_model_repo
        )

        if arch is None:
            raise ValueError(
                f"No architecture found for {self.model_config.huggingface_model_repo.repo_id}. "
                "This should not happen after config resolution."
            )

        # Get pipeline task
        task = PIPELINE_REGISTRY.retrieve_pipeline_task(self)

        # Format weight_path the same way as log_pipeline_info
        if len(self.model_config.weight_path) == 1:
            # Single weight path - keep it inline
            weight_path = f"{self.model_config.weight_path[0]}"
        else:
            # Multiple weight paths - format as comma-separated list
            weight_path = ", ".join(
                f"{path}" for path in self.model_config.weight_path
            )

        # Get reserved memory info from KVCache config
        kv_config = self.model_config._kv_cache_config
        memory_str = "Not calculated"
        if kv_config._available_cache_memory is not None:
            cache_gb = kv_config._available_cache_memory / (1024**3)
            memory_str = f"{cache_gb:.2f} GB"

        # Log basic configuration
        logger.info("")
        logger.info(
            "Pipeline Configuration (use --pretty-print-config to print full config)"
        )
        logger.info("=" * 70)
        logger.info(f"    model_name:         {arch.name}")
        logger.info(f"    task:               {task}")
        logger.info(f"    weight_path:        {weight_path}")
        logger.info(f"    max_batch_size:     {self.max_batch_size}")
        logger.info(f"    max_seq_len:        {self.max_length}")
        logger.info(f"    cache_memory:       {memory_str}")
        logger.info("")

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "max_length": "Set the maximum sequence length for input data processed by the model. This must be less than the value specified in the Hugging Face configuration file. The default is derived from the Hugging Face configuration value. Larger values may consume more memory.",
            "pipeline_role": "Whether the pipeline should serve both a prefill or decode role or both.",
            "max_batch_size": "Define the maximum batch size to execute with the model. When not specified (None), we determine this value dynamically. For users launching in a server scenario, the expectation is that this value should be set higher based on server capacity.",
            "max_ce_batch_size": "Set the maximum cache size reserved for a single context encoding batch. The effective limit will be the lesser of this value and `max-batch-size`. Default is 192.",
            "max_queue_size_tg": "Maximum number of requests in decode queue. By default, this is max-batch-size.",
            "min_batch_size_tg": "Specifies a soft floor on the decode batch size. If the TG batch size is larger than this value, the scheduler will continue to run TG batches. If it falls below, the scheduler will prioritize CE. This is an experimental flag solely for the TTS scheduler.",
            "ce_delay_ms": "Duration of scheduler sleep prior to starting a prefill batch. This is an experimental flag solely for the TTS scheduler. Default is 0.0.",
            "enable_prioritize_first_decode": "When enabled, the scheduler will always run a TG batch immediately after a CE batch, with the same requests. This may be useful for decreasing time-to-first-chunk latency. This is an experimental flag solely for the TTS scheduler. Default is false.",
            "enable_chunked_prefill": "Enable chunked prefill to split context encoding requests into multiple chunks based on `prefill-chunk-size`. Default is true.",
            "enable_in_flight_batching": "When enabled, prioritizes token generation by batching it with context encoding requests. Default is false.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is -1 which specifies a default value based on configuration and platform. Ignored for models which are not auto-regressive (e.g. embedding models).",
            "prefill_chunk_size": "The target number of un-encoded tokens to include in each batch. This value is used for chunked prefill and memory estimation. Default is 8192.",
            "enable_echo": "Whether the model should be built with echo capabilities. This defaults to false.",
            "pool_embeddings": "Whether to pool embedding outputs. Default is true.",
            "use_experimental_kernels": "Whether to use experimental kernels. Default is false.",
            "pdl_level": "Level of overlap of kernel launch via programmatic dependent grid control. Default is 0.",
            "custom_architectures": "A list of custom architecture implementations to register. Each input can either be a raw module name or an import path followed by a colon and the module name.",
        }

    @property
    def model_config(self) -> MAXModelConfig:
        return self._model_config

    @property
    def draft_model_config(self) -> Optional[MAXModelConfig]:
        return self._draft_model_config

    @property
    def sampling_config(self) -> SamplingConfig:
        return self._sampling_config

    @property
    def profiling_config(self) -> ProfilingConfig:
        return self._profiling_config

    @property
    def lora_config(self) -> Optional[LoRAConfig]:
        return self._lora_config


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


@dataclass
class AudioGenerationConfig(PipelineConfig):
    # TODO: Make these flags more discoverable.
    audio_decoder: str = ""
    """The name of the audio decoder model architecture."""

    audio_decoder_weights: str = ""
    """The path to the audio decoder weights file."""

    chunk_size: list[int] | None = None
    """The chunk sizes to use for streaming.
    If this is an int, then fixed-size chunks of the given size are used
    If this is a list, then variable chunk sizes are used."""

    buffer: int = 0
    """The number of previous speech tokens to pass to the audio decoder on
    each generation step."""

    block_causal: bool = False
    """Whether prior buffered tokens should attend to tokens in the current block.
    Has no effect if buffer is not set."""

    prepend_prompt_speech_tokens: PrependPromptSpeechTokens = (
        PrependPromptSpeechTokens.ONCE
    )
    """Whether the prompt speech tokens should be forwarded to the audio decoder.
    If "never", the prompt tokens are not forwarded.
    If "once", the prompt tokens are only forwarded on the first block.
    If "always", the prompt tokens are forwarded on all blocks.
    """

    prepend_prompt_speech_tokens_causal: bool = False
    """Whether the prompt speech tokens should attend to tokens in the currently
    generated audio block.
    Has no effect if prepend_prompt_speech_tokens is "never".
    If False (default), the prompt tokens do not attend to the current block.
    If True, the prompt tokens attend to the current block.
    """

    audio_decoder_config: dict[str, Any] = field(default_factory=dict)
    """Parameters to pass to the audio decoder model."""

    _run_model_test_mode: bool = False
    """Test-only flag that indicates that test parameters have been passed to
    the model, such as leaving the audio decoder weights empty or using a
    dummy speech language model."""

    prometheus_metrics_mode: PrometheusMetricsMode = (
        PrometheusMetricsMode.INSTRUMENT_ONLY
    )
    """The mode to use for Prometheus metrics."""

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
