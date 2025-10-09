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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import numpy.typing as npt
from max.driver import load_devices
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    PipelineTask,
    PipelineTokenizer,
    TextGenerationRequest,
)
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import TextAndVisionContext, TextContext
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

PipelineTypes = Union[  # noqa: UP007 (This breaks a mypy check, unsure why)
    TextGenerationPipeline[TextContext],
    EmbeddingsPipeline,
    AudioGeneratorPipeline,
    SpeculativeDecodingTextGenerationPipeline,
    SpeechTokenGenerationPipeline,
]


def get_pipeline_for_task(
    task: PipelineTask, pipeline_config: PipelineConfig
) -> (
    type[TextGenerationPipeline[TextContext]]
    | type[EmbeddingsPipeline]
    | type[SpeculativeDecodingTextGenerationPipeline]
    | type[AudioGeneratorPipeline]
    | type[SpeechTokenGenerationPipeline]
):
    if task == PipelineTask.TEXT_GENERATION:
        if pipeline_config.draft_model_config is not None:
            return SpeculativeDecodingTextGenerationPipeline
        else:
            return TextGenerationPipeline[TextContext]
    elif task == PipelineTask.EMBEDDINGS_GENERATION:
        return EmbeddingsPipeline
    elif task == PipelineTask.AUDIO_GENERATION:
        return AudioGeneratorPipeline
    elif task == PipelineTask.SPEECH_TOKEN_GENERATION:
        return SpeechTokenGenerationPipeline
    else:
        raise ValueError(
            f"PipelineTask ({task}) does not have supported Pipeline"
        )


@dataclass(frozen=False)
class SupportedArchitecture:
    """
    Represents a model architecture configuration for MAX pipelines.

    This class defines all the necessary components and settings required to
    support a specific model architecture within the MAX pipeline system.
    Each `SupportedArchitecture` instance encapsulates the model implementation,
    tokenizer, supported encodings, and other architecture-specific configuration.

    New architectures should be registered into the :obj:`PipelineRegistry`
    using the :obj:`~PipelineRegistry.register()` method.

    Example:
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
                rope_type=RopeType.none,
                weight_adapters={
                    WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
                    # Add other weight formats if needed
                },
                multi_gpu_supported=True,  # Set based on your implementation capabilities
                required_arguments={"some_arg": True},
                task=PipelineTask.TEXT_GENERATION,
            )
    """

    name: str
    """The name of the model architecture that must match the Hugging Face model class name."""

    example_repo_ids: list[str]
    """A list of Hugging Face repository IDs that use this architecture for testing and validation purposes."""

    default_encoding: SupportedEncoding
    """The default quantization encoding to use when no specific encoding is requested."""

    supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]]
    """A dictionary mapping supported quantization encodings to their compatible KV cache strategies."""

    pipeline_model: type[PipelineModel[Any]]
    """The `PipelineModel` class that defines the model graph structure and execution logic."""

    task: PipelineTask
    """The pipeline task type that this architecture supports."""

    tokenizer: Callable[..., PipelineTokenizer[Any, Any, Any]]
    """A callable that returns a `PipelineTokenizer` instance for preprocessing model inputs."""

    default_weights_format: WeightsFormat
    """The weights format expected by the `pipeline_model`."""

    rope_type: RopeType = RopeType.none
    """The type of RoPE (Rotary Position Embedding) used by the model."""

    weight_adapters: dict[WeightsFormat, WeightsAdapter] = field(
        default_factory=dict
    )
    """A dictionary of weight format adapters for converting checkpoints from different formats to the default format."""

    multi_gpu_supported: bool = False
    """Whether the architecture supports multi-GPU execution."""

    required_arguments: dict[str, bool | int | float] = field(
        default_factory=dict
    )
    """A dictionary specifying required values for PipelineConfig options."""

    context_validators: list[
        Callable[[TextContext | TextAndVisionContext], None]
    ] = field(default_factory=list)
    """A list of callable context validators for the architecture."""

    @property
    def tokenizer_cls(self) -> type[PipelineTokenizer[Any, Any, Any]]:
        if isinstance(self.tokenizer, type):
            return self.tokenizer
        # Otherwise fall back to PipelineTokenizer.
        return TextTokenizer


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
                raise ValueError(
                    f"Refusing to override existing architecture for '{architecture.name}'"
                )
            logger.warning(
                f"Overriding existing architecture for '{architecture.name}'"
            )

        self.architectures[architecture.name] = architecture

    def retrieve_architecture(
        self, huggingface_repo: HuggingFaceRepo
    ) -> SupportedArchitecture | None:
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

    def retrieve_tokenizer(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
    ) -> PipelineTokenizer[Any, Any, Any]:
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

        tokenizer: PipelineTokenizer[Any, Any, Any]
        if (
            arch.pipeline_model.__name__ in ("MistralModel", "Phi3Model")
            and arch.tokenizer is TextTokenizer
        ):
            text_tokenizer = cast(type[TextTokenizer], arch.tokenizer)
            tokenizer = text_tokenizer(
                pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=pipeline_config.retrieve_chat_template(),
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                pipeline_config=pipeline_config,
                chat_template=pipeline_config.retrieve_chat_template(),
            )

        return tokenizer

    def retrieve_factory(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer[Any, Any, Any], Callable[[], PipelineTypes]]:
        tokenizer: PipelineTokenizer[Any, Any, Any]
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
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=pipeline_config.retrieve_chat_template(),
                context_validators=arch.context_validators,
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                pipeline_config=pipeline_config,
                chat_template=pipeline_config.retrieve_chat_template(),
                context_validators=arch.context_validators,
            )
        # Cast tokenizer to the proper type for text generation pipeline compatibility
        typed_tokenizer = cast(
            PipelineTokenizer[
                Any, npt.NDArray[np.integer[Any]], TextGenerationRequest
            ],
            tokenizer,
        )
        pipeline_factory = cast(
            Callable[[], PipelineTypes],
            functools.partial(
                pipeline_class,
                pipeline_config=pipeline_config,
                pipeline_model=arch.pipeline_model,
                eos_token_id=tokenizer.eos,
                weight_adapters=arch.weight_adapters,
                tokenizer=typed_tokenizer,
            ),
        )

        if tokenizer.eos is None:
            raise ValueError(
                "tokenizer.eos value is None, tokenizer configuration is incomplete."
            )

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
    ) -> tuple[PipelineTokenizer[Any, Any, Any], PipelineTypes]:
        tokenizer, pipeline_factory = self.retrieve_factory(
            pipeline_config, task, override_architecture
        )
        return tokenizer, pipeline_factory()

    def reset(self) -> None:
        self.architectures.clear()


PIPELINE_REGISTRY = PipelineRegistry([])
