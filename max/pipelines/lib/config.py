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
from typing import Any, Optional, get_type_hints

from max.driver import DeviceSpec, load_devices
from max.graph.quantization import QuantizationEncoding

from .config_enums import PipelineEngine, PipelineRole
from .max_config import (
    KVCacheConfig,
    MAXConfig,
    ProfilingConfig,
    SamplingConfig,
)
from .memory_estimation import MEMORY_ESTIMATOR
from .model_config import MAXModelConfig
from .registry import PIPELINE_REGISTRY

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

    max_length: Optional[int] = None
    """Maximum sequence length of the model."""

    max_new_tokens: int = -1
    """Maximum number of new tokens to generate during a single inference pass of the model."""

    pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode
    """Whether the pipeline should serve both a prefill or decode role or both."""

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
    encoding requests."""

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

    pool_embeddings: bool = True
    """Whether to pool embedding outputs."""

    use_experimental_kernels: str = os.environ.get(
        "USE_EXPERIMENTAL_KERNELS", "false"
    )

    pdl_level: str = os.environ.get("PDL_LEVEL", "1")
    """Level of overlap of kernel launch via programmatic dependent grid control."""

    ignore_eos: bool = False
    """Ignore EOS and continue generating tokens, even when an EOS variable is hit."""

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

    def __init__(self, **kwargs: Any) -> None:
        # Initialize all fields with their defaults first
        for curr_field in fields(self.__class__):
            if curr_field.default is not MISSING:
                setattr(self, curr_field.name, curr_field.default)
            elif curr_field.default_factory is not MISSING:
                setattr(self, curr_field.name, curr_field.default_factory())

        # Check against draft model first
        draft_kwargs = {}
        for k, v in list(kwargs.items()):
            if k.startswith("draft"):
                field_name = k.replace("draft_", "")
                if field_name in MAXModelConfig.__dataclass_fields__:
                    draft_kwargs[field_name] = v
                    del kwargs[k]

        if draft_kwargs.get("model_path", "") != "":
            self._draft_model_config = MAXModelConfig(**draft_kwargs)
        else:
            self._draft_model_config = None

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

                        if self._draft_model_config:
                            self._draft_model_config._kv_cache_config = (
                                KVCacheConfig(**kv_cache_kwargs)
                            )

                    elif config_name == "_sampling_config" and (
                        self.enable_echo or self._draft_model_config
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

        # NOTE: Do not use this directly after instantiating PipelineConfig. We
        # only keep this here to support backward compatibility of the draft_model
        # field entrypoint. This will be removed entirely soon. I purposefully
        # set this to an empty string than None, to ensure that we catch any
        # inadvertent use of draft_model.
        self.draft_model = ""
        if unmatched_kwargs:
            raise ValueError(f"Unmatched kwargs: {unmatched_kwargs}")

        self.resolve()

    def resolve(self) -> None:
        """
        Validates and resolves the config.

        This method is called after the config is initialized, to ensure that all
        config fields have been initialized to a valid state.
        """
        # Before anything else, import custom model modules to add them to the registry.
        for module_spec in self.custom_architectures:
            module_parts = module_spec.split(":")
            if len(module_parts) > 2:
                msg = (
                    "Custom module spec contains too many colons: {module_spec}"
                )
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

        self.model_config.resolve()
        # Validate if a provided max_length is non-negative.
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("max_length must be non-negative.")

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

        if (
            self.sampling_config.frequency_penalty != 0
            or self.sampling_config.presence_penalty != 0
            or self.sampling_config.repetition_penalty != 1
        ):
            if self.draft_model_config:
                raise ValueError(
                    "frequency_penalty, presence_penalty and repetition_penalty are not currently supported with speculative decoding."
                )
            if self.sampling_config.repetition_penalty <= 0:
                raise ValueError("repetition_penalty must be greater than 0.")
            self.sampling_config.do_penalties = True

        # Run Baseline Validation
        self._validate_and_resolve_remaining_pipeline_config(self.model_config)

        # Run Additional Checks for Speculative Decoding
        if self.draft_model_config:
            self._validate_and_resolve_remaining_pipeline_config(
                self.draft_model_config
            )

            self._validate_pipeline_config_for_speculative_decoding()

    def _validate_pipeline_config_for_speculative_decoding(self) -> None:
        """
        Validate the pipeline configs when used in speculative decoding mode.
        """
        assert self.draft_model_config is not None  # keep mypy happy

        # We don't support running speculative decoding with the HuggingFace backend.
        if self.engine == PipelineEngine.HUGGINGFACE:
            msg = (
                "Speculative Decoding not supported with the HuggingFace Engine"
            )
            raise ValueError(msg)

        # Validate that both the `draft_model` and target model `model_path` have the same
        # architecture
        draft_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.draft_model_config.huggingface_model_repo,
        )

        if not draft_arch:
            msg = "MAX-Optimized architecture not found for `draft_model`"
            raise ValueError(msg)

        target_arch = PIPELINE_REGISTRY.retrieve_architecture(
            huggingface_repo=self.model_config.huggingface_model_repo,
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
            huggingface_repo=model_config.huggingface_model_repo,
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
                f" '{model_config.model_path}' falling back to"
                " HuggingFace."
            )
            logger.warning(msg)
            msg = "Please file a request at https://modul.ar/request to add this model architecture to MAX."
            logger.warning(msg)
            self.engine = PipelineEngine.HUGGINGFACE
            return

        model_config.validate_multi_gpu_supported(
            multi_gpu_supported=arch.multi_gpu_supported
        )

        # The remainder of this function, assumes we have both a valid model_path,
        # and a SupportedArchitecture. We should then validate the details of the existing architecture
        # and fallback to HuggingFace if needed.

        model_config.validate_and_resolve_quantization_encoding_weight_path(
            default_encoding=arch.default_encoding
        )

        model_config.validate_and_resolve_rope_type(
            arch_rope_type=arch.rope_type
        )

        # by this point, the quantization_encoding must be provided. verify it is supported.
        if model_config.quantization_encoding not in arch.supported_encodings:
            if self.engine == PipelineEngine.MAX:
                msg = f"quantization_encoding of '{model_config.quantization_encoding}' not supported by MAX engine, unable to run with engine = 'max'."
                raise ValueError(msg)

            else:
                msg = f"quantization_encoding of '{model_config.quantization_encoding}' not supported by MAX engine, falling back to HuggingFace."
                logger.warning(msg)
                self.engine = PipelineEngine.HUGGINGFACE
                return

        model_config.validate_and_resolve_with_set_quantization_encoding(
            supported_encodings=arch.supported_encodings,
            default_weights_format=arch.default_weights_format,
        )

        devices = load_devices(model_config.device_specs)
        MEMORY_ESTIMATOR.estimate_memory_footprint(
            self, arch.pipeline_model, model_config, devices
        )

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
            "max_length": "Set the maximum sequence length for input data processed by the model. This must be less than the value specified in the Hugging Face configuration file. The default is derived from the Hugging Face configuration value. Larger values may consume more memory.",
            "max_new_tokens": "Specify the maximum number of new tokens to generate during a single inference pass of the model. Default is -1, which means the model will generate until the maximum sequence length is hit, or and eos token is generated.",
            "max_batch_size": "Define the maximum cache size reserved for a single batch. This value defaults to 1. Increase this value based on server capacity when deploying in production.",
            "max_ce_batch_size": "Set the maximum cache size reserved for a single context encoding batch. The effective limit will be the lesser of this value and `max-batch-size`. Default is 32.",
            "pipeline_role": "Whether the pipeline should serve both a prefill or decode role or both.",
            "enable_chunked_prefill": "Enable chunked prefill to split context encoding requests into multiple chunks based on `target-num-new-tokens`",
            "enable_in_flight_batching": "When enabled, prioritizes token generation by batching it with context encoding requests.",
            "rope_type": "Force using a specific rope type, `none` | `normal' | `nexo`. Only matters for GGUF weights.",
            "max_num_steps": "Specify the number of steps to run for multi-step scheduling during inference. Default is set to 1.",
            "pad_to_multiple_of": "Pad input tensors to be a multiple of value provided. Default is set to 2.",
            "enable_echo": "Whether the model should be built with echo capabilities. This defaults to false.",
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
    def draft_model_config(self) -> Optional[MAXModelConfig]:
        return self._draft_model_config

    @property
    def sampling_config(self) -> SamplingConfig:
        return self._sampling_config

    @property
    def profiling_config(self) -> ProfilingConfig:
        return self._profiling_config


@dataclass
class AudioGenerationConfig(PipelineConfig):
    # TODO: Make these flags more discoverable.
    audio_decoder: str = ""
    """The name of the audio decoder model architecture."""
    audio_prompt_speakers: str = ""
    """The path to the audio prompt speakers file."""
    audio_decoder_weights: str = ""
    """The path to the audio decoder weights file."""
    block_sizes: list[int] | int | None = None
    """The block sizes to use for streaming.
    If this is an int, then fixed-size blocks of the given size are used
    If this is a list, then variable block sizes are used."""
    buffered: bool = False
    """Whether to pass through all generated LLM tokens so far to the audio decoder
    (after the first block)."""
    block_causal: bool = False
    """Whether prior buffered blocks should attend to tokens in the current block.
    Has no effect if buffered is False."""
    prepend_prompt_speech_tokens: bool | None = None
    """Whether the prompt speech tokens should be forwarded to the audio decoder.
    If None (default), the prompt tokens are not forwarded.
    If False, the prompt tokens are only forwarded on the first block.
    If True, the prompt tokens are forwarded on all blocks.
    """
    prepend_prompt_speech_tokens_causal: bool = False
    """Whether the prompt speech tokens should attend to tokens in the currently
    generated audio block.
    Has no effect if prepend_prompt_speech_tokens is False.
    If False (default), the prompt tokens do not attend to the current block.
    If True, the prompt tokens attend to the current block.
    """

    def __init__(self, audio_config: dict[str, str], **kwargs: Any) -> None:
        PipelineConfig.__init__(self, **kwargs)

        self.audio_decoder = audio_config.pop("audio_decoder", "")
        if not self.audio_decoder:
            raise ValueError(
                "When running the audio generation task, --audio-decoder must be specified"
            )
        self.audio_prompt_speakers = audio_config.pop(
            "audio_prompt_speakers", ""
        )
        self.audio_decoder_weights = audio_config.pop(
            "audio_decoder_weights", ""
        )

        # Configuration for audio generation streaming.
        buffered = audio_config.pop("buffered", "")
        if not buffered or buffered.lower() == "false":
            self.buffered = False
        else:
            self.buffered = True

        block_causal = audio_config.pop("block_causal", "")
        if not block_causal or block_causal.lower() == "false":
            self.block_causal = False
        else:
            self.block_causal = True

        prepend_prompt_speech_tokens = audio_config.pop(
            "prepend_prompt_speech_tokens", ""
        )
        if not prepend_prompt_speech_tokens:
            self.prepend_prompt_speech_tokens = None
        elif prepend_prompt_speech_tokens.lower() == "false":
            self.prepend_prompt_speech_tokens = False
        else:
            self.prepend_prompt_speech_tokens = True

        prepend_prompt_speech_tokens_causal = audio_config.pop(
            "prepend_prompt_speech_tokens_causal", ""
        )
        if (
            not prepend_prompt_speech_tokens_causal
            or prepend_prompt_speech_tokens_causal.lower() == "false"
        ):
            self.prepend_prompt_speech_tokens_causal = False
        else:
            self.prepend_prompt_speech_tokens_causal = True

        if self.block_sizes:
            raise NotImplementedError("Block sizes are not implemented")
        if self.buffered:
            raise NotImplementedError("Buffered generation is not implemented")
        if self.block_causal:
            raise NotImplementedError("Causal generation is not implemented")
        if self.prepend_prompt_speech_tokens:
            raise NotImplementedError(
                "Prepend prompt tokens is not implemented"
            )
        if self.prepend_prompt_speech_tokens_causal:
            raise NotImplementedError(
                "Prepend prompt speech tokens causal is not implemented"
            )

        if audio_config:
            raise ValueError(
                f"Unknown audio generation option(s): {audio_config}"
            )
