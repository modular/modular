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

"""Model registry, for tracking various model variants."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

from max.driver import Device, load_devices
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import EmbeddingsGenerator, PipelineTask, PipelineTokenizer
from max.nn.kv_cache import KVCacheStrategy
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from .audio_generator_pipeline import AudioGeneratorPipeline
    from .config import PipelineConfig

from .audio_generator_pipeline import AudioGeneratorPipeline
from .config_enums import RopeType, SupportedEncoding
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import HuggingFaceRepo
from .pipeline import PipelineModel, TextGenerationPipeline
from .speculative_decoding import SpeculativeDecodingTextGenerationPipeline
from .speech_token_pipeline import SpeechTokenGenerationPipeline
from .tokenizer import TextTokenizer

logger = logging.getLogger("max.pipelines")

PipelineTypes = Union[
    TextGenerationPipeline,
    EmbeddingsGenerator,
    AudioGeneratorPipeline,
    SpeculativeDecodingTextGenerationPipeline,
    SpeechTokenGenerationPipeline,
]


def get_pipeline_for_task(
    task: PipelineTask, pipeline_config: PipelineConfig
) -> (
    type[TextGenerationPipeline]
    | type[EmbeddingsPipeline]
    | type[SpeculativeDecodingTextGenerationPipeline]
    | type[AudioGeneratorPipeline]
    | type[SpeechTokenGenerationPipeline]
):
    if task == PipelineTask.TEXT_GENERATION:
        if pipeline_config.draft_model_config is not None:
            return SpeculativeDecodingTextGenerationPipeline
        else:
            return TextGenerationPipeline
    elif task == PipelineTask.EMBEDDINGS_GENERATION:
        return EmbeddingsPipeline
    elif task == PipelineTask.AUDIO_GENERATION:
        return AudioGeneratorPipeline
    elif task == PipelineTask.SPEECH_TOKEN_GENERATION:
        return SpeechTokenGenerationPipeline
    else:
        msg = f"PipelineTask ({task}) does not have supported Pipeline"
        raise ValueError(msg)


class SupportedArchitecture:
    def __init__(
        self,
        name: str,
        example_repo_ids: list[str],
        default_encoding: SupportedEncoding,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
        pipeline_model: type[PipelineModel],
        task: PipelineTask,
        tokenizer: Callable[..., PipelineTokenizer],
        default_weights_format: WeightsFormat,
        multi_gpu_supported: bool = False,
        rope_type: RopeType = RopeType.none,
        weight_adapters: dict[WeightsFormat, WeightsAdapter] | None = None,
    ) -> None:
        """Represents a model architecture configuration for MAX pipelines.

        This class defines all the necessary components and settings required to
        support a specific model architecture within the MAX pipeline system.
        Each `SupportedArchitecture` instance encapsulates the model implementation,
        tokenizer, supported encodings, and other architecture-specific configuration.

        New architectures should be registered into the :obj:`PipelineRegistry`
        using the :obj:`~PipelineRegistry.register()` method.

        .. code-block:: python

            my_architecture = SupportedArchitecture(
                name="MyModelForCausalLM",  # Must match your Hugging Face model class name
                example_repo_ids=[
                    "your-org/your-model-name",  # Add example model repository IDs
                ],
                default_encoding=SupportedEncoding.q4_k,
                supported_encodings={
                    SupportedEncoding.q4_k: [KVCacheStrategy.PAGED],
                    SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
                    # Add other encodings your model supports
                },
                pipeline_model=MyModel,
                tokenizer=TextTokenizer,
                default_weights_format=WeightsFormat.safetensors,
                multi_gpu_supported=True,  # Set based on your implementation capabilities
                weight_adapters={
                    WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
                    # Add other weight formats if needed
                },
                task=PipelineTask.TEXT_GENERATION,
            )

        Args:
            name: The name of the model architecture that must match the Hugging Face
                model class name.
            example_repo_ids: A list of Hugging Face repository IDs that use this
                architecture for testing and validation purposes.
            default_encoding: The default quantization encoding to use when no
                specific encoding is requested.
            supported_encodings: A dictionary mapping supported quantization encodings
                to their compatible KV cache strategies.
            pipeline_model: The `PipelineModel` class that defines the model graph
                structure and execution logic.
            task: The pipeline task type that this architecture supports.
            tokenizer: A callable that returns a `PipelineTokenizer` instance for
                preprocessing model inputs.
            default_weights_format: The weights format expected by the `pipeline_model`.
            multi_gpu_supported: Whether the architecture supports multi-GPU execution.
            rope_type: The type of RoPE (Rotary Position Embedding) used by the model.
            weight_adapters: A dictionary of weight format adapters for converting
                checkpoints from different formats to the default format.
        """
        self.name = name
        self.example_repo_ids = example_repo_ids
        self.default_encoding = default_encoding
        self.supported_encodings = supported_encodings
        self.pipeline_model = pipeline_model
        self.tokenizer = tokenizer
        self.default_weights_format = default_weights_format
        self.multi_gpu_supported = multi_gpu_supported
        self.rope_type = rope_type
        self.weight_adapters = weight_adapters or {}
        self.task = task

    def __eq__(self, other: Any) -> bool:
        if other.__class__ == self.__class__:
            for field in [
                "name",
                "example_repo_ids",
                "default_encoding",
                "supported_encodings",
                "pipeline_model",
                "tokenizer",
                "default_weights_format",
                "rope_type",
                "weight_adapters",
                "task",
            ]:
                if not (hasattr(other, field) and hasattr(self, field)):
                    return False

                if getattr(other, field) != getattr(self, field):
                    return False

            return True

        return False

    @property
    def tokenizer_cls(self) -> type[PipelineTokenizer]:
        if isinstance(self.tokenizer, type):
            return self.tokenizer
        # Otherwise fall back to PipelineTokenizer.
        return PipelineTokenizer


class PipelineRegistry:
    def __init__(self, architectures: list[SupportedArchitecture]) -> None:
        self.architectures = {arch.name: arch for arch in architectures}
        self._cached_huggingface_configs: dict[HuggingFaceRepo, AutoConfig] = {}
        self._cached_huggingface_tokenizers: dict[
            HuggingFaceRepo, PreTrainedTokenizer | PreTrainedTokenizerFast
        ] = {}

    def register(
        self,
        architecture: SupportedArchitecture,
        *,
        allow_override: bool = False,
    ) -> None:
        """Add new architecture to registry."""
        if architecture.name in self.architectures:
            if not allow_override:
                msg = f"Refusing to override existing architecture for '{architecture.name}'"
                raise ValueError(msg)
            logger.warning(
                f"Overriding existing architecture for '{architecture.name}'"
            )

        self.architectures[architecture.name] = architecture

    def retrieve_architecture(
        self, huggingface_repo: HuggingFaceRepo
    ) -> Optional[SupportedArchitecture]:
        # Retrieve model architecture names
        hf_config = self.get_active_huggingface_config(
            huggingface_repo=huggingface_repo
        )
        architecture_names = getattr(hf_config, "architectures", [])

        if not architecture_names:
            logger.debug(
                "architectures not listed in HuggingFace config, cannot be matched against MAX Registry"
            )
            return None

        for architecture_name in architecture_names:
            if architecture_name in self.architectures:
                return self.architectures[architecture_name]

        logger.debug(
            f"optimized architecture not available for {huggingface_repo.repo_id} in MAX REGISTRY"
        )

        return None

    def get_active_huggingface_config(
        self, huggingface_repo: HuggingFaceRepo
    ) -> AutoConfig:
        """Retrieves or creates a cached HuggingFace AutoConfig for the given
        model configuration.

        This method maintains a cache of HuggingFace configurations to avoid
        reloading them unnecessarily which incurs a huggingface hub API call.
        If a config for the given model hasn't been loaded before, it will
        create a new one using AutoConfig.from_pretrained() with the model's
        settings.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            AutoConfig: The HuggingFace configuration object for the model.
        """
        # TODO: This is a hack to get around the fact that in serving and the
        # way we instantiate multiprocess model workers, pickling AutoConfig will
        # not work and AutoConfig.from_pretrained will need to be called again
        # when trust_remote_code=True.
        if huggingface_repo.trust_remote_code:
            return AutoConfig.from_pretrained(
                huggingface_repo.repo_id,
                trust_remote_code=huggingface_repo.trust_remote_code,
                revision=huggingface_repo.revision,
            )

        if huggingface_repo not in self._cached_huggingface_configs:
            self._cached_huggingface_configs[huggingface_repo] = (
                AutoConfig.from_pretrained(
                    huggingface_repo.repo_id,
                    trust_remote_code=huggingface_repo.trust_remote_code,
                    revision=huggingface_repo.revision,
                )
            )

        return self._cached_huggingface_configs[huggingface_repo]

    def get_active_tokenizer(
        self, huggingface_repo: HuggingFaceRepo
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Retrieves or creates a cached HuggingFace AutoTokenizer for the given
        model configuration.

        This method maintains a cache of HuggingFace tokenizers to avoid
        reloading them unnecessarily which incurs a huggingface hub API call.
        If a tokenizer for the given model hasn't been loaded before, it will
        create a new one using AutoTokenizer.from_pretrained() with the model's
        settings.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The HuggingFace tokenizer for the model.
        """
        if huggingface_repo not in self._cached_huggingface_tokenizers:
            self._cached_huggingface_tokenizers[huggingface_repo] = (
                AutoTokenizer.from_pretrained(
                    huggingface_repo.repo_id,
                    trust_remote_code=huggingface_repo.trust_remote_code,
                    revision=huggingface_repo.revision,
                )
            )

        return self._cached_huggingface_tokenizers[huggingface_repo]

    def _load_logging_message(
        self,
        pipeline_config: PipelineConfig,
        tokenizer_type: type[PipelineTokenizer],
        pipeline_name: str,
        pipeline_model: str,
        factory: bool,
        devices: list[Device],
        architecture_id: Optional[str] = None,
    ):
        weight_path = ",\n        ".join(
            [
                f"                               {path}"
                for path in pipeline_config.model_config.weight_path
            ]
        )
        factory_str = "factory" if factory else ""

        weights_repo_str = (
            f"\n            weights_repo_id:        {pipeline_config.model_config._weights_repo_id}"
            if pipeline_config.model_config._weights_repo_id
            else ""
        )

        devices_str = ", ".join(f"{d.label}[{d.id}]" for d in devices)

        quantization_encoding_str = str(
            pipeline_config.model_config.quantization_encoding
        )
        if pipeline_config.model_config._applied_dtype_cast_from:
            quantization_encoding_str = f"{quantization_encoding_str} (cast from {pipeline_config.model_config._applied_dtype_cast_from})"

        message = f"""

        Loading {tokenizer_type.__name__} and {pipeline_name}({pipeline_model}) {factory_str} for:
            architecture:           {architecture_id if architecture_id else "UNKNOWN"}
            devices:                {devices_str}
            model_path:             {pipeline_config.model_config.model_path}{weights_repo_str}
            huggingface_revision:   {pipeline_config.model_config.huggingface_model_revision}
            quantization_encoding:  {quantization_encoding_str}
            cache_strategy:         {pipeline_config.model_config.kv_cache_config.cache_strategy}
            weight_path:            [
        {weight_path}
                                    ]
        """

        return message

    def retrieve_tokenizer(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
    ) -> PipelineTokenizer:
        """Retrieves a tokenizer for the given pipeline configuration.

        Args:
            pipeline_config: Configuration for the pipeline
            override_architecture: Optional architecture override string

        Returns:
            PipelineTokenizer: The configured tokenizer

        Raises:
            ValueError: If no architecture is found
        """
        # MAX pipeline
        arch: SupportedArchitecture | None = None
        if override_architecture:
            arch = self.architectures[override_architecture]
        else:
            arch = self.retrieve_architecture(
                huggingface_repo=pipeline_config.model_config.huggingface_model_repo
            )

        if arch is None:
            raise ValueError(
                f"No architecture found for {pipeline_config.model_config.huggingface_model_repo.repo_id}"
            )

        # Calculate Max Length
        huggingface_config = pipeline_config.model_config.huggingface_config
        max_length = arch.pipeline_model.calculate_max_seq_len(
            pipeline_config, huggingface_config=huggingface_config
        )

        tokenizer: PipelineTokenizer
        if (
            arch.pipeline_model.__name__ in ("MistralModel", "Phi3Model")
            and arch.tokenizer is TextTokenizer
        ):
            text_tokenizer = cast(type[TextTokenizer], arch.tokenizer)
            tokenizer = text_tokenizer(
                pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=max_length,
                max_new_tokens=pipeline_config.max_new_tokens,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                enable_llama_whitespace_fix=True,
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=max_length,
                max_new_tokens=pipeline_config.max_new_tokens,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                pipeline_config=pipeline_config,
            )

        return tokenizer

    def retrieve_factory(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer, Callable[[], PipelineTypes]]:
        tokenizer: PipelineTokenizer
        pipeline_factory: Callable[[], PipelineTypes]

        pipeline_class = get_pipeline_for_task(task, pipeline_config)

        # MAX pipeline
        arch: SupportedArchitecture | None = None
        if override_architecture:
            arch = self.architectures[override_architecture]
        else:
            arch = self.retrieve_architecture(
                huggingface_repo=pipeline_config.model_config.huggingface_model_repo
            )

        # Load HuggingFace Config
        huggingface_config = pipeline_config.model_config.huggingface_config

        # Architecture should not be None here, as the engine is MAX.
        assert arch is not None
        devices = load_devices(pipeline_config.model_config.device_specs)
        logger.info(
            self._load_logging_message(
                pipeline_config=pipeline_config,
                tokenizer_type=arch.tokenizer_cls,
                pipeline_model=arch.pipeline_model.__name__,
                pipeline_name=pipeline_class.__name__,
                architecture_id=arch.name,
                factory=True,
                devices=devices,
            )
        )

        max_length = arch.pipeline_model.calculate_max_seq_len(
            pipeline_config, huggingface_config=huggingface_config
        )

        # Old Mistral model like Mistral-7B-Instruct-v0.3 uses LlamaTokenizer
        # and suffers from the whitespace decoding bug. So, we enable the fix
        # for only MistralModel in order to avoid any issues with performance
        # for rest of the models. This can be applied more generically once
        # we have more time verifying this for all the models.
        # More information:
        # https://linear.app/modularml/issue/AIPIPE-197/add-support-for-mistral-7b-instruct-v03
        # TODO: remove this pipeline_model.__name__ check
        if (
            arch.pipeline_model.__name__ in ("MistralModel", "Phi3Model")
            and arch.tokenizer is TextTokenizer
        ):
            text_tokenizer = cast(type[TextTokenizer], arch.tokenizer)
            tokenizer = text_tokenizer(
                pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=max_length,
                max_new_tokens=pipeline_config.max_new_tokens,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                enable_llama_whitespace_fix=True,
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=max_length,
                max_new_tokens=pipeline_config.max_new_tokens,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                pipeline_config=pipeline_config,
            )
        pipeline_factory = cast(
            Callable[[], PipelineTypes],
            functools.partial(
                pipeline_class,
                pipeline_config=pipeline_config,
                pipeline_model=arch.pipeline_model,
                eos_token_id=tokenizer.eos,
                weight_adapters=arch.weight_adapters,
            ),
        )

        if tokenizer.eos is None:
            msg = "tokenizer.eos value is None, tokenizer configuration is incomplete."
            raise ValueError(msg)

        return tokenizer, pipeline_factory

    def retrieve_pipeline_task(
        self, pipeline_config: PipelineConfig
    ) -> PipelineTask:
        """
        Retrieve the pipeline task associated with the architecture for the given pipeline configuration.

        Args:
            pipeline_config (PipelineConfig): The configuration for the pipeline.

        Returns:
            PipelineTask: The task associated with the architecture.

        Raises:
            ValueError: If no supported architecture is found for the given model repository.
        """
        if arch := self.retrieve_architecture(
            huggingface_repo=pipeline_config.model_config.huggingface_model_repo
        ):
            return arch.task

        raise ValueError(
            f"MAX Optimized architecture not supported for {pipeline_config.model_config.huggingface_model_repo.repo_id}"
        )

    def retrieve(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer, PipelineTypes]:
        tokenizer, pipeline_factory = self.retrieve_factory(
            pipeline_config, task, override_architecture
        )
        return tokenizer, pipeline_factory()

    def reset(self) -> None:
        self.architectures.clear()


PIPELINE_REGISTRY = PipelineRegistry([])
