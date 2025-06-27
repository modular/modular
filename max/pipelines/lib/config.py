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

from .config_enums import PipelineEngine, PipelineRole
from .lora import LoRAManager
from .max_config import (
    KVCacheConfig,
    LoRAConfig,
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

    pdl_level: str = os.environ.get("PDL_LEVEL", "0")
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

    _lora_config: Optional[LoRAConfig] = None
    """The LoRA config."""

    _lora_manager: Optional[LoRAManager] = None
    """The LoRA Manager"""

    def __init__(self, **kwargs: Any) -> None:
        # Initialize all fields with their defaults first
        for curr_field in fields(self.__class__):
            if curr_field.default is not MISSING:
                setattr(self, curr_field.name, curr_field.default)
            elif curr_field.default_factory is not MISSING:
                setattr(self, curr_field.name, curr_field.default_factory())

        lora_kwargs = {}
        for k, v in list(kwargs.items()):
            if k in LoRAConfig.__dataclass_fields__:
                lora_kwargs[k] = v
                del kwargs[k]

        if lora_kwargs.get("lora_paths", []):
            self._lora_config = LoRAConfig(**lora_kwargs)
        else:
            self._lora_config = None

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

    def _import_custom_architectures(self) -> None:
        """
        Import custom model modules to add them to the registry.
        """
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

        # We don't support running speculative decoding with the HuggingFace backend.
        if self.engine == PipelineEngine.HUGGINGFACE:
            msg = (
                "Speculative Decoding not supported with the HuggingFace Engine"
            )
            raise ValueError(msg)

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
            if self.engine == PipelineEngine.MAX:
                msg = f"quantization_encoding of '{model_config.quantization_encoding}' not supported by MAX engine, unable to run with engine = 'max'."
                raise ValueError(msg)

            else:
                msg = f"quantization_encoding of '{model_config.quantization_encoding}' not supported by MAX engine, falling back to HuggingFace."
                logger.warning(msg)
                self.engine = PipelineEngine.HUGGINGFACE
                return

        model_config.validate_and_resolve_with_resolved_quantization_encoding(
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

    @property
    def lora_config(self) -> Optional[LoRAConfig]:
        return self._lora_config

    @property
    def lora_manager(self) -> Optional[LoRAManager]:
        return self._lora_manager


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

    block_sizes: list[int] | None = None
    """The block sizes to use for streaming.
    If this is an int, then fixed-size blocks of the given size are used
    If this is a list, then variable block sizes are used."""

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
        block_sizes: list[int] | None = None,
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
        self.block_sizes = block_sizes
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
        audio_decoder = audio_flags.pop("audio_decoder", "")
        if not audio_decoder:
            raise ValueError(
                "When running the audio generation task, --audio-decoder must be specified"
            )
        audio_decoder_weights = audio_flags.pop("audio_decoder_weights", "")

        # Configuration for audio generation streaming.
        block_sizes_str = audio_flags.pop("block_sizes", "")
        if not block_sizes_str:
            block_sizes = None
        else:
            block_sizes = [int(size) for size in block_sizes_str.split(",")]

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
            block_sizes=block_sizes,
            buffer=buffer,
            block_causal=block_causal,
            prepend_prompt_speech_tokens=prepend_prompt_speech_tokens,
            prepend_prompt_speech_tokens_causal=prepend_prompt_speech_tokens_causal,
            run_model_test_mode=run_model_test_mode,
            prometheus_metrics_mode=prometheus_metrics_mode,
            **config_flags,
        )
