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

import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Optional, get_type_hints

from max.driver import DeviceSpec, load_devices
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightsFormat, weights_format
from max.pipelines.config_enums import (
    _ALTERNATE_ENCODINGS,
    PipelineEngine,
    RopeType,
    SupportedEncoding,
)
from max.pipelines.kv_cache import KVCacheStrategy
from max.pipelines.max_config import (
    KVCacheConfig,
    MAXConfig,
    MAXModelConfig,
    ProfilingConfig,
    SamplingConfig,
    repo_exists_with_retry,
)
from max.pipelines.registry import PIPELINE_REGISTRY
from transformers import AutoTokenizer

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=False)
class PipelineConfig(MAXConfig):
    """Configuration for a pipeline.

    WIP - Once a PipelineConfig is fully initialized, it should be as immutable
    as possible (frozen=True). All underlying dataclass fields should have been
    initialized to their default values, be it user specified via some CLI
    flag, config file, environment variable, or internally set to a reasonable
    default.
    """

    engine: Optional[PipelineEngine] = None
    """Engine backend to use for serving, 'max' for the max engine, or 'huggingface' as fallback option for improved model coverage."""

    serialized_model_path: Optional[str] = None
    """DEPRECATED: Serialization paths no longer supported."""

    save_to_serialized_model_path: Optional[str] = None
    """DEPRECATED: Serialization paths no longer supported."""

    max_length: Optional[int] = None
    """Maximum sequence length of the model."""

    max_new_tokens: int = -1
    """Maximum number of new tokens to generate during a single inference pass of the model."""

    max_batch_size: Optional[int] = None
    """Maximum batch size to execute with the model.
    This is set to one, to minimize memory consumption for the base case, in which a person is
    running a local server to test out MAX. For users launching in a server scenario, the expectation
    is that this value should be set higher based on server capacity."""

    max_ce_batch_size: int = 192
    """Maximum cache size to reserve for a single context encoding batch.
    The actual limit is the lesser of this and `max_batch_size`."""

    enable_chunked_prefill: bool = True
    """Enable chunked prefill to split context encoding requests into multiple chunks
    based on 'target_num_new_tokens'."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context
    encoding requests. Requires chunked prefill."""

    max_num_steps: int = -1
    """The number of steps to run for multi-step scheduling. -1 specifies a default value based on
    configuration and platform. Ignored for models which are not auto-regressive (e.g. embedding
    models)."""

    pad_to_multiple_of: int = 2
    """Pad input tensors to be a multiple of value provided."""

    target_num_new_tokens: Optional[int] = None
    """The target number of un-encoded tokens to include in each batch.
    If not set, this will be set to a best-guess optimal value based on model, hardware, and available memory."""

    enable_echo: bool = False
    """Whether the model should be built with echo capabilities."""

    rope_type: Optional[RopeType] = None
    """Force using a specific rope type: `none` | `normal` | `neox`. Only matters for GGUF weights."""

    pool_embeddings: bool = True
    """Whether to pool embedding outputs."""

    max_cache_batch_size: Optional[int] = None
    """DEPRECATED: The maximum cache batch size to use for the model. Use max_batch_size instead."""

    use_experimental_kernels: str = os.environ.get(
        "USE_EXPERIMENTAL_KERNELS", "false"
    )

    draft_model: Optional[str] = None
    """Draft model for use during Speculative Decoding."""

    ignore_eos: bool = False
    """Ignore EOS and continue generating tokens, even when an EOS variable is hit."""

    _model_config: MAXModelConfig = field(default_factory=MAXModelConfig)
    """The model config."""

    _sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    """The sampling config."""

    _profiling_config: ProfilingConfig = field(default_factory=ProfilingConfig)

    """The profiling config."""

    def __init__(self, **kwargs: Any) -> None:
        # Initialize all fields with their defaults first
        for curr_field in fields(self.__class__):
            if curr_field.default is not MISSING:
                setattr(self, curr_field.name, curr_field.default)
            elif curr_field.default_factory is not MISSING:
                setattr(self, curr_field.name, curr_field.default_factory())

        # Check if any kwargs are meant for other MAXConfig classes
        unmatched_kwargs: dict[str, Any] = {}
        # Then process kwargs which override defaults
        for key, value in list(kwargs.items()):
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
                del kwargs[key]
            else:
                unmatched_kwargs[key] = value

        # Try to match unmatched kwargs with other config classes
        if unmatched_kwargs:
            # TODO(zheng): Make this more efficient by using MaxConfig instance
            # instead of hardcoding the config names.
            for config_name in [
                "_sampling_config",
                "_profiling_config",
                # TODO(zheng): Remove this once backward compatibility is no
                # longer needed for MAXModelConfig.
                "_model_config",
            ]:
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
                    if config_name == "_model_config" and kv_cache_kwargs:
                        # Create new model config with updated KVCache config
                        model_config = config_class(**matched_kwargs)
                        model_config._kv_cache_config = KVCacheConfig(
                            **kv_cache_kwargs
                        )
                        setattr(self, config_name, model_config)
                    elif config_name == "_sampling_config" and (
                        self.enable_echo or self.draft_model
                    ):
                        sampling_config = config_class(**matched_kwargs)
                        sampling_config.enable_variable_logits = True
                        setattr(self, config_name, sampling_config)
                    else:
                        setattr(
                            self, config_name, config_class(**matched_kwargs)
                        )

                    # Remove matched kwargs
                    for key in matched_kwargs:
                        del unmatched_kwargs[key]
                    for key in kv_cache_kwargs:
                        del unmatched_kwargs[key]

        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        self.validate()

    def validate(self) -> None:
        """
        Validate the config.

        This method is called after the config is initialized, to ensure that all
        config fields have been initialized to a valid state.
        """
        self._model_config.validate()

        # Validate if a provided max_length is non-negative.
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("max_length must be non-negative.")

        if self.max_cache_batch_size is not None:
            logger.warning(
                "--max-cache-batch-size is deprecated, use `--max-batch-size` instead. This setting will stop working in a future release."
            )
            self.max_batch_size = self.max_cache_batch_size

        # Set sensible defaults. These are platform-specific.
        if self.max_num_steps < 0:
            if (
                self.sampling_config.enable_structured_output
                or self.model_config.device_specs[0] == DeviceSpec.cpu()
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

        if self.sampling_config.enable_structured_output:
            if self.model_config.device_specs[0] == DeviceSpec.cpu():
                raise ValueError(
                    "enable_structured_output is not currently supported on CPU."
                )

        # Run Baseline Validation
        self._validate_pipeline_config()

        if self.draft_model:
            if not repo_exists_with_retry(self.draft_model):
                raise ValueError(
                    "draft_model provided does not exist on HuggingFace."
                    "Only public HuggingFace draft models currently supported."
                )

            # Run Additional Checks for Speculative Decoding
            self._validate_pipeline_config_for_speculative_decoding()

    def _validate_pipeline_config_for_speculative_decoding(self) -> None:
        """
        Validate the pipeline configs when used in speculative decoding mode.
        """
        assert self.draft_model is not None, "draft_model must be provided"

        # Assume `draft_model` is provided, and thus speculative decoding is enabled.
        # We don't support running speculative decoding with the HuggingFace backend.
        if self.engine == PipelineEngine.HUGGINGFACE:
            msg = (
                "Speculative Decoding not supported with the HuggingFace Engine"
            )
            raise ValueError(msg)

        # Validate that both the `draft_model` and target model `model_path` have the same
        # architecture
        draft_arch = PIPELINE_REGISTRY.retrieve_architecture(
            self.draft_model,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        if not draft_arch:
            msg = "MAX-Optimized architecture not found for `draft_model`"
            raise ValueError(msg)

        target_arch = PIPELINE_REGISTRY.retrieve_architecture(
            self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if not target_arch:
            msg = "MAX-Optimized architecture not found for target model (`model_path`)"
            raise ValueError(msg)

        if draft_arch != target_arch:
            msg = f"architecture for the draft_model ({draft_arch.name}) does not match the architecture retrieved for the target model ({target_arch.name})"
            raise ValueError(msg)

        # Validate that their tokenizers are identical.
        draft_tokenizer = AutoTokenizer.from_pretrained(
            self.draft_model,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        target_tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        # Compare Vocabularies
        if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
            msg = f"tokenizer for draft_model ({self.draft_model}) does not match the vocabulary of the tokenizer for the target model ({self.model_config.model_path})"
            raise ValueError(msg)

        # Compare Tokenizer Configuration
        if draft_tokenizer.__dict__ == target_tokenizer.__dict__:
            msg = f"tokenizer for draft_model ({self.draft_model}) does not match the configuration of the tokenizer for the target model ({self.model_config.model_path})"
            raise ValueError(msg)

    def _validate_pipeline_config(self) -> None:
        """Update pipeline config with appropriate values if not provided.
        If invalid config is provided, error out with detailed reason."""
        # Retrieve the architecture
        arch = PIPELINE_REGISTRY.retrieve_architecture(
            model_path=self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        # If nothing is provided, we should not update any more params.
        # Instead, fall back to the HuggingFace engine.
        if not arch and self.engine == PipelineEngine.MAX:
            raise ValueError(
                "MAX-optimized architecture not available, failing as engine is provided as 'MAX'"
            )

        elif not arch:
            msg = (
                "MAX-optimized architecture not available for"
                f" '{self.model_config.model_path}' falling back to"
                " HuggingFace."
            )
            logger.warning(msg)
            self.engine = PipelineEngine.HUGGINGFACE
            return

        if (
            not arch.multi_gpu_supported
            and len(self.model_config.device_specs) > 1
            and self.model_config.device_specs[0].device_type == "gpu"
        ):
            raise ValueError(
                f"Multiple GPU inference is currently not supported for {self.model_config.model_path}."
            )

        # The remainder of this function, assumes we have both a valid model_path,
        # and a SupportedArchitecture. We should then validate the details of the existing architecture
        # and fallback to HuggingFace if needed.

        # If weight_path and quantization_encoding are provided, verify that they are consistent.
        huggingface_weights_repo = self.model_config.huggingface_weights_repo()
        try:
            _weights_format = weights_format(self.model_config.weight_path)
        except ValueError:
            _weights_format = None
        if (
            self.model_config.weight_path
            and self.model_config.quantization_encoding
            # Cannot validate quantization_encoding for pytorch.
            and _weights_format != WeightsFormat.pytorch
        ):
            # Get the encoding of the first weight path file.
            if os.path.exists(self.model_config.weight_path[0]):
                file_encoding = SupportedEncoding.parse_from_file_name(
                    str(self.model_config.weight_path[0])
                )
            else:
                file_encoding = huggingface_weights_repo.encoding_for_file(
                    self.model_config.weight_path[0]
                )

            if file_encoding:
                if file_encoding != self.model_config.quantization_encoding:
                    msg = f"weight_path provided '{self.model_config.weight_path[0]}' has an inconsistent encoding '{file_encoding}' than quantization_encoding provided '{self.model_config.quantization_encoding}'. Please update one."
                    raise ValueError(msg)
        # If weight path is not None, infer the quantization_encoding from the weight_path.
        elif (
            self.model_config.weight_path
            and not self.model_config.quantization_encoding
            and _weights_format != WeightsFormat.pytorch
        ):
            if os.path.exists(self.model_config.weight_path[0]):
                # Not currently supported. Infer encoding from local path.
                if self.model_config.weight_path[0].suffix == ".safetensors":
                    msg = "If a local safetensors file is provided, please provide a quantization_encoding."
                    raise ValueError(msg)

                if encoding := SupportedEncoding.parse_from_file_name(
                    str(self.model_config.weight_path[0])
                ):
                    msg = f"encoding inferred from weights file: {encoding}"
                    logger.debug(msg)
                    self.model_config.quantization_encoding = encoding

            else:
                if encoding := huggingface_weights_repo.encoding_for_file(
                    self.model_config.weight_path[0]
                ):
                    msg = f"encoding inferred from weights file: {encoding}"
                    logger.debug(msg)
                    self.model_config.quantization_encoding = encoding
                else:
                    msg = f"encoding cannot be inferred from weights file: {self.model_config.weight_path[0]}, please pass a quantization_encoding explictly."
                    raise ValueError(msg)
        elif not self.model_config.quantization_encoding:
            # Check if the repo only has one quantization_encoding.
            supported_encodings = huggingface_weights_repo.supported_encodings
            if len(supported_encodings) == 1:
                msg = f"huggingface repo only has '{supported_encodings[0]}' weights, using '{supported_encodings[0]}'"
                logger.debug(msg)
                self.model_config.quantization_encoding = supported_encodings[0]
            elif (
                not self.model_config.device_specs[0].device_type == "cpu"
            ) and SupportedEncoding.bfloat16 in arch.supported_encodings:
                # TODO(AITLIB-137): replace this with more full featured logic.
                # If we are running on an accelerator and the quantiziation encoding is not set, override to bfloat16.
                self.model_config.quantization_encoding = (
                    SupportedEncoding.bfloat16
                )
            else:
                msg = f"encoding not provided, using default encoding of {arch.default_encoding}"
                logger.debug(msg)
                self.model_config.quantization_encoding = arch.default_encoding
        # by this point, the quantization_encoding must be provided. verify it is supported.
        if (
            self.model_config.quantization_encoding
            not in arch.supported_encodings
        ):
            if self.engine == PipelineEngine.MAX:
                msg = f"quantization_encoding of '{self.model_config.quantization_encoding}' not supported by MAX engine, unable to run with engine = 'max'."
                raise ValueError(msg)

            else:
                msg = f"quantization_encoding of '{self.model_config.quantization_encoding}' not supported by MAX engine, falling back to HuggingFace."
                logger.warning(msg)
                self.engine = PipelineEngine.HUGGINGFACE
                return

        # Check that the quantization encoding is supported on the specified
        # devices.
        for device_spec in self.model_config.device_specs:
            if not self.model_config.quantization_encoding.supported_on(
                device_spec
            ):
                available_encodings = list(arch.supported_encodings.keys())

                msg = (
                    f"The encoding '{self.model_config.quantization_encoding}' is not compatible with the selected device type '{device_spec.device_type}'.\n\n"
                    f"You have two options to resolve this:\n"
                    f"1. Use a different device\n"
                    f"2. Use a different encoding (encodings available for this model: {', '.join(str(enc) for enc in available_encodings)})\n\n"
                    f"Please use the --help flag for more information."
                )

                raise ValueError(msg)

        self.model_config.finalize_encoding_config()

        # We should now have a valid quantization_encoding, and possibly a weight_path.
        # If no weight_path is provided, we should grab the default.
        if not self.model_config.weight_path:
            # Retrieve the default files for each weights format.

            # Get alternate encoding (e.g. if float32 is requested and there are
            # only bfloat16 weights, allow retrieving the bfloat16 weights
            # because they can be cast to float32).
            if self.model_config.quantization_encoding:
                alternate_encoding = _ALTERNATE_ENCODINGS.get(
                    self.model_config.quantization_encoding
                )
            else:
                alternate_encoding = None

            weight_files = huggingface_weights_repo.files_for_encoding(
                encoding=self.model_config.quantization_encoding,
                alternate_encoding=alternate_encoding,
            )

            if default_weight_files := weight_files.get(
                arch.default_weights_format, []
            ):
                self.model_config.weight_path = default_weight_files
            elif weight_files:
                # Load any available weight file.
                self.model_config.weight_path = next(
                    iter(weight_files.values())
                )

        if not self.model_config.weight_path:
            if self.model_config.quantization_encoding not in [
                SupportedEncoding.bfloat16,
                SupportedEncoding.float32,
            ]:
                msg = f"compatible weights cannot be found for '{self.model_config.quantization_encoding}' in 'gguf' format, in the provided repo: '{huggingface_weights_repo.repo_id}'"
                raise ValueError(msg)
            else:
                msg = f"compatible weights cannot be found for '{self.model_config.quantization_encoding}'"
                raise ValueError(msg)

        # Check supported_cache_strategy
        supported_cache_strategies = arch.supported_encodings.get(
            self.model_config.quantization_encoding, []
        )
        if (
            self.model_config.kv_cache_config.cache_strategy
            == KVCacheStrategy.MODEL_DEFAULT
            and supported_cache_strategies
        ):
            default_strategy = supported_cache_strategies[0]
            msg = f"default cache_strategy of '{default_strategy}' enabled"
            logger.debug(msg)

            self.model_config.kv_cache_config.cache_strategy = default_strategy
        elif (
            supported_cache_strategies
            and self.model_config.kv_cache_config.cache_strategy
            not in supported_cache_strategies
        ):
            supported_strategy = supported_cache_strategies[0]

            msg = f"cache_strategy = '{self.model_config.kv_cache_config.cache_strategy}' not supported for '{self.model_config.quantization_encoding}', using '{supported_strategy}' cache strategy."
            logger.warning(msg)

            self.model_config.kv_cache_config.cache_strategy = (
                supported_strategy
            )

        # Assume at this point, an architecture,
        # a model_path and weight_paths are available.
        assert self.model_config.weight_path, "weight_path must be provided."
        for path in self.model_config.weight_path:
            # Check if file exists locally.
            if not os.path.exists(path):
                # If does not exist locally, verify that it exists on Huggingface.
                if not huggingface_weights_repo.file_exists(str(path)):
                    msg = (
                        f"weight_path: '{path}' does not exist locally, and"
                        f" '{self.model_config.model_path}/{path}' does"
                        " not exist on HuggingFace."
                    )
                    raise ValueError(msg)

        if self.rope_type is None:
            self.rope_type = arch.rope_type

        devices = load_devices(self.model_config.device_specs)
        PIPELINE_REGISTRY._estimate_memory_footprint(self, arch, devices)

        # If we pass validation ensure and the engine is not set, just set it
        # to MAX.
        if self.engine is None:
            self.engine = PipelineEngine.MAX

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

    @staticmethod
    def help() -> dict[str, str]:
        pipeline_help = {
            "engine": "Specify the engine backend to use for serving the model. Options include `max` for the MAX engine, or `huggingface` as a fallback option that provides improved model coverage.",
            "weight_path": "Provide an optional local path or path relative to the root of a Hugging Face repo to the model weights you want to use. This allows you to specify custom weights instead of using defaults. You may pass multiple, ie. `--weight-path=model-00001-of-00002.safetensors --weight-path=model-00002-of-00002.safetensors`",
            "serialized_model_path": "DEPRECATED: Serialization paths no longer supported.",
            "save_to_serialized_model_path": "DEPRECATED: Serialization paths no longer supported.",
            "max_length": "Set the maximum sequence length for input data processed by the model. This must be less than the value specified in the Hugging Face configuration file. The default is derived from the Hugging Face configuration value. Larger values may consume more memory.",
            "max_new_tokens": "Specify the maximum number of new tokens to generate during a single inference pass of the model. Default is -1, which means the model will generate until the maximum sequence length is hit, or and eos token is generated.",
            "max_batch_size": "Define the maximum cache size reserved for a single batch. This value defaults to 1. Increase this value based on server capacity when deploying in production.",
            "max_ce_batch_size": "Set the maximum cache size reserved for a single context encoding batch. The effective limit will be the lesser of this value and `max-cache-batch-size`. Default is 32.",
            "enable_chunked_prefill": "Enable chunked prefill to split context encoding requests into multiple chunks based on `target-num-new-tokens`",
            "enable_in_flight_batching": "When enabled, prioritizes token generation by batching it with context encoding requests. Requires chunked prefill.",
            "max_cache_batch_size": "DEPRECATED: Use `max_batch_size` instead.",
            "rope_type": "Force using a specific rope type, `none` | `normal' | `nexo`. Only matters for GGUF weights.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is set to 1.",
            "pad_to_multiple_of": "Pad input tensors to be a multiple of value provided. Default is set to 2.",
            "enable_echo": "Whether the model should be built with echo capabilities. This defaults to false.",
            "draft_model": "Draft model for use in speculative decoding.",
            "ignore_eos": "Ignore EOS and continue generating tokens, even when an EOS variable is hit.",
        }

        # Add help text for all MAX config classes
        # TODO(zheng): Make this more efficient by using MaxConfig instance
        # instead of hardcoding the config names.
        for config_class in [SamplingConfig, ProfilingConfig]:
            config_help = config_class.help()  # type: ignore
            for key in config_help:
                if key in pipeline_help:
                    raise ValueError(
                        f"Duplicate help key '{key}' found in {config_class.__name__}"
                    )
            pipeline_help.update(config_help)
        return pipeline_help

    @property
    def model_config(self) -> MAXModelConfig:
        return self._model_config

    @property
    def sampling_config(self) -> SamplingConfig:
        return self._sampling_config

    @property
    def profiling_config(self) -> ProfilingConfig:
        return self._profiling_config
