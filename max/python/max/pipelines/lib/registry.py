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

"""Model registry, for tracking various model variants."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from huggingface_hub import model_info
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    EmbeddingsContext,
    Pipeline,
    PipelineTask,
    PipelineTokenizer,
    TextGenerationContext,
    TextGenerationRequest,
)
from max.nn.legacy.kv_cache import KVCacheStrategy
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
from .config_enums import PipelineRole, RepoType, RopeType, SupportedEncoding
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import HuggingFaceRepo, is_diffusion_pipeline
from .interfaces import ArchConfig, ArchConfigWithKVCache, PipelineModel
from .pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from .pipeline_variants.pixel_generation import PixelGenerationPipeline
from .pipeline_variants.text_generation import TextGenerationPipeline
from .speculative_decoding import (
    EAGLESpeculativeDecodingPipeline,
    StandaloneSpeculativeDecodingPipeline,
)
from .speech_token_pipeline import SpeechTokenGenerationPipeline
from .tokenizer import TextTokenizer

logger = logging.getLogger("max.pipelines")

PipelineTypes: TypeAlias = Pipeline[Any, Any]


def _infer_task_from_hf_pipeline_tag(
    pipeline_tag: str | None,
) -> PipelineTask | None:
    """Map Hugging Face pipeline tag to MAX PipelineTask.

    Args:
        pipeline_tag: The pipeline tag from Hugging Face Hub model info.

    Returns:
        The corresponding PipelineTask or None if no mapping exists.
    """
    if pipeline_tag is None:
        return None

    # Map HF pipeline tags to MAX tasks
    tag_to_task = {
        "text-generation": PipelineTask.TEXT_GENERATION,
        "feature-extraction": PipelineTask.EMBEDDINGS_GENERATION,
        "sentence-similarity": PipelineTask.EMBEDDINGS_GENERATION,
        "audio-generation": PipelineTask.AUDIO_GENERATION,
        "text-to-image": PipelineTask.PIXEL_GENERATION,
        "image-to-image": PipelineTask.PIXEL_GENERATION,
        # Add more mappings as needed
    }

    return tag_to_task.get(pipeline_tag)


def get_pipeline_for_task(
    task: PipelineTask, pipeline_config: PipelineConfig
) -> (
    type[TextGenerationPipeline[TextContext]]
    | type[EmbeddingsPipeline]
    | type[AudioGeneratorPipeline]
    | type[PixelGenerationPipeline[Any]]
    | type[StandaloneSpeculativeDecodingPipeline]
    | type[SpeechTokenGenerationPipeline]
    | type[EAGLESpeculativeDecodingPipeline]
    | type[OverlapTextGenerationPipeline[TextContext]]
):
    """Returns the pipeline class for the given task and config.

    Args:
        task: The pipeline task (e.g. text generation, embeddings).
        pipeline_config: Pipeline configuration (may select speculative path).

    Returns:
        The pipeline class to use for this task and config.
    """
    if (
        task == PipelineTask.TEXT_GENERATION
        and pipeline_config.speculative is not None
    ):
        spec_method = pipeline_config.speculative.speculative_method
        assert spec_method is not None
        if pipeline_config.enable_overlap_scheduler:
            raise ValueError(
                "Overlap scheduler is not supported with speculative decoding yet."
            )

        if pipeline_config.speculative.is_standalone():
            return StandaloneSpeculativeDecodingPipeline
        elif (
            pipeline_config.speculative.is_eagle()
            or pipeline_config.speculative.is_mtp()
        ):
            return EAGLESpeculativeDecodingPipeline
        else:
            raise ValueError(f"Unsupported speculative method: {spec_method}")
    elif pipeline_config.enable_overlap_scheduler:
        role = pipeline_config.pipeline_role
        if (
            task == PipelineTask.TEXT_GENERATION
            and role == PipelineRole.PrefillAndDecode
        ):
            return OverlapTextGenerationPipeline[TextContext]
        raise ValueError(
            "Overlap scheduler is only supported for TEXT_GENERATION task "
            f"and PrefillAndDecode pipeline role, got {task} and {role}"
        )
    elif task == PipelineTask.TEXT_GENERATION:
        return TextGenerationPipeline[TextContext]
    elif task == PipelineTask.EMBEDDINGS_GENERATION:
        return EmbeddingsPipeline
    elif task == PipelineTask.AUDIO_GENERATION:
        return AudioGeneratorPipeline
    elif task == PipelineTask.SPEECH_TOKEN_GENERATION:
        return SpeechTokenGenerationPipeline
    elif task == PipelineTask.PIXEL_GENERATION:
        return PixelGenerationPipeline


@dataclass(frozen=False)
class SupportedArchitecture:
    """Represents a model architecture configuration for MAX pipelines.

    Defines the components and settings required to
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
                context_type=TextContext,
                config=MyModelConfig,  # Architecture-specific config class
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

    context_type: type[TextGenerationContext] | type[EmbeddingsContext]
    """The context class type that this architecture uses for managing request state and inputs.

    This should be a class (not an instance) that implements either the `TextGenerationContext`
    or `EmbeddingsContext` protocol, defining how the pipeline processes and tracks requests.
    """

    config: type[ArchConfig]
    """The architecture-specific configuration class for the model.

    This class must implement the :obj:`ArchConfig` protocol, providing an
    :obj:`initialize` method that creates a configuration instance from a
    :obj:`PipelineConfig`. For models with KV cache, this should be a class
    implementing :obj:`ArchConfigWithKVCache` to enable KV cache memory estimation.
    """

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
    """A list of callable validators that verify context inputs before model execution.

    These validators are called during context creation to ensure inputs meet
    model-specific requirements. Validators should raise `InputError` for invalid
    inputs, providing early error detection before expensive model operations.

    .. code-block:: python

        def validate_single_image(context: TextContext | TextAndVisionContext) -> None:
            if isinstance(context, TextAndVisionContext):
                if context.pixel_values and len(context.pixel_values) > 1:
                    raise InputError(f"Model supports only 1 image, got {len(context.pixel_values)}")

        my_architecture = SupportedArchitecture(
            # ... other fields ...
            context_validators=[validate_single_image],
        )
    """

    supports_empty_batches: bool = False
    """Whether the architecture can handle empty batches during inference.

    When set to True, the pipeline can process requests with zero-sized batches
    without errors. This is useful for certain execution modes and expert parallelism.
    Most architectures do not require empty batch support and should leave this as False.
    """

    requires_max_batch_context_length: bool = False
    """Whether the architecture requires a max batch context length to be specified.

    If True and max_batch_context_length is not specified, we will default to
    the max sequence length of the model.
    """

    @property
    def tokenizer_cls(self) -> type[PipelineTokenizer[Any, Any, Any]]:
        """Returns the tokenizer class for this architecture."""
        if isinstance(self.tokenizer, type):
            return self.tokenizer
        # Otherwise fall back to PipelineTokenizer.
        return TextTokenizer


class PipelineRegistry:
    """Registry for managing supported model architectures and their pipelines.

    This class maintains a collection of :class:`SupportedArchitecture`
    instances, each defining how a particular model architecture should be
    loaded, configured, and executed.

    .. note::

        Do not instantiate this class directly. Always use the global
        :obj:`PIPELINE_REGISTRY` singleton, which is automatically populated
        with all built-in architectures when you import :mod:`max.pipelines`.

    Use :obj:`PIPELINE_REGISTRY` when you want to:

    - **Register a custom architectures**: Call :meth:`register` to add a new
      MAX model architecture to the registry before loading it.
    - **Query supported models**: Call :meth:`retrieve_architecture` to check
      if a Hugging Face model repository is supported before attempting to load it.
    - **Access cached configs**: Methods like :meth:`get_active_huggingface_config` and
      :meth:`get_active_tokenizer` provide cached access to model configurations and tokenizers.
    """

    def __init__(self, architectures: list[SupportedArchitecture]) -> None:
        # Primary lookup by architecture name
        self.architectures = {arch.name: arch for arch in architectures}
        # Secondary lookup for architectures with duplicate names, keyed by (name, task)
        self._architectures_by_task: dict[
            tuple[str, PipelineTask], SupportedArchitecture
        ] = {}
        self._cached_huggingface_configs: dict[HuggingFaceRepo, AutoConfig] = {}
        self._cached_diffusers_configs: dict[
            HuggingFaceRepo, dict[str, Any] | None
        ] = {}
        self._cached_huggingface_tokenizers: dict[
            HuggingFaceRepo, PreTrainedTokenizer | PreTrainedTokenizerFast
        ] = {}

    def register(
        self,
        architecture: SupportedArchitecture,
        *,
        allow_override: bool = False,
    ) -> None:
        """Add new architecture to registry.

        If multiple architectures share the same name but have different tasks,
        they are registered in a secondary lookup table keyed by (name, task).
        """
        task_key = (architecture.name, architecture.task)

        if architecture.name in self.architectures:
            existing_arch = self.architectures[architecture.name]

            # If same task, this is a true conflict
            if existing_arch.task == architecture.task:
                if not allow_override:
                    raise ValueError(
                        f"Refusing to override existing architecture for '{architecture.name}' "
                        f"with task {architecture.task}"
                    )
                logger.warning(
                    f"Overriding existing architecture for '{architecture.name}' with task {architecture.task}"
                )
                self.architectures[architecture.name] = architecture
                self._architectures_by_task[task_key] = architecture
            else:
                # Different tasks - store both, using task-based lookup
                logger.info(
                    f"Registering multiple architectures with name '{architecture.name}': "
                    f"{existing_arch.task} and {architecture.task}"
                )
                # Move existing arch to task-based lookup if not already there
                existing_key = (existing_arch.name, existing_arch.task)
                if existing_key not in self._architectures_by_task:
                    self._architectures_by_task[existing_key] = existing_arch
                # Add new arch to task-based lookup
                self._architectures_by_task[task_key] = architecture
        else:
            # First registration of this name
            self.architectures[architecture.name] = architecture
            self._architectures_by_task[task_key] = architecture

    def retrieve_architecture(
        self,
        huggingface_repo: HuggingFaceRepo,
        use_legacy_module: bool = True,
        task: PipelineTask | None = None,
    ) -> SupportedArchitecture | None:
        """Retrieve architecture matching the Hugging Face model config.

        Args:
            huggingface_repo: The Hugging Face repository to match against.
            use_legacy_module: Whether to use legacy Module architecture (default=True).
                When True, appends "_Legacy" suffix to find legacy graph-based architecture.
                When False, uses the standard Hugging Face architecture name for new API.
            task: Optional task to disambiguate when multiple architectures share the same name.
                  If not provided and multiple architectures share the same name, the task will
                  be inferred from the Hugging Face Hub's pipeline_tag.

        Returns:
            The matching SupportedArchitecture or None if no match found.
        """
        # Retrieve model architecture names
        is_diffusion = is_diffusion_pipeline(huggingface_repo)

        if not is_diffusion:
            hf_config = self.get_active_huggingface_config(
                huggingface_repo=huggingface_repo
            )
            architecture_names = getattr(hf_config, "architectures", [])
        else:
            diffusers_config = self.get_active_diffusers_config(
                huggingface_repo=huggingface_repo
            )
            if diffusers_config is None:
                logger.warning(
                    f"Could not load diffusers config (model_index.json) for {huggingface_repo.repo_id}. "
                    f"This may be a gated model requiring authentication. "
                    f"Please set HF_TOKEN environment variable or run 'huggingface-cli login'."
                )
                return None
            if diffusers_arch := diffusers_config.get("_class_name"):
                architecture_names = [diffusers_arch]
            else:
                logger.debug(
                    f"No `_class_name` found in diffusers_config for {huggingface_repo.repo_id}"
                )
                return None

        if not architecture_names:
            logger.debug(
                "architectures not listed in HuggingFace config, cannot be matched against MAX Registry"
            )
            return None

        for architecture_name in architecture_names:
            if use_legacy_module:
                architecture_name += "_Legacy"

            # If task not provided, check if we need to infer it
            inferred_task = task
            if task is None:
                # Check if multiple architectures share this name
                matching_tasks = [
                    arch_task
                    for arch_name, arch_task in self._architectures_by_task
                    if arch_name == architecture_name
                ]

                # If multiple architectures share the name, infer task from pipeline_tag
                if len(matching_tasks) > 1:
                    try:
                        hf_model_info = model_info(
                            huggingface_repo.repo_id,
                            revision=huggingface_repo.revision,
                        )
                        inferred_task = _infer_task_from_hf_pipeline_tag(
                            hf_model_info.pipeline_tag
                        )
                        if inferred_task is not None:
                            logger.debug(
                                f"Inferred task '{inferred_task}' from pipeline_tag "
                                f"'{hf_model_info.pipeline_tag}' for {huggingface_repo.repo_id}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to infer task from HuggingFace Hub: {e}. "
                            f"Using first registered architecture."
                        )

            if arch := self._resolve_architecture(
                architecture_name, inferred_task
            ):
                return arch

        logger.debug(
            f"optimized architecture not available for {huggingface_repo.repo_id} in MAX REGISTRY"
        )

        return None

    def get_active_huggingface_config(
        self, huggingface_repo: HuggingFaceRepo
    ) -> AutoConfig:
        """Retrieves or creates a cached Hugging Face AutoConfig for the given model.

        Maintains a cache of Hugging Face configurations to avoid
        reloading them unnecessarily which incurs a Hugging Face Hub API call.
        If a config for the given model hasn't been loaded before, it will
        create a new one using AutoConfig.from_pretrained() with the model's
        settings.

        Note: The cache key (HuggingFaceRepo) includes trust_remote_code in its
        hash, so configs with different trust settings are cached separately.
        For multiprocessing, each worker process has its own registry instance
        with an empty cache, so configs are loaded fresh in each worker.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            AutoConfig: The Hugging Face configuration object for the model.
        """
        if huggingface_repo not in self._cached_huggingface_configs:
            self._cached_huggingface_configs[huggingface_repo] = (
                AutoConfig.from_pretrained(
                    huggingface_repo.repo_id,
                    trust_remote_code=huggingface_repo.trust_remote_code,
                    revision=huggingface_repo.revision,
                )
            )

        return self._cached_huggingface_configs[huggingface_repo]

    def get_active_diffusers_config(
        self, huggingface_repo: HuggingFaceRepo
    ) -> dict[str, Any] | None:
        """Retrieves or creates a cached diffusers config for the given repository.

        This method checks if the repository is a diffusion pipeline by looking for
        model_index.json. If found, it downloads and caches the config. If not found,
        returns None.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            dict | None: The diffusers config dict if this is a diffusion pipeline, None otherwise.
        """
        if huggingface_repo not in self._cached_diffusers_configs:
            try:
                # Check if model_index.json exists to identify diffusion pipelines
                import json

                if huggingface_repo.repo_type == RepoType.local:
                    config_path = os.path.join(
                        huggingface_repo.repo_id, "model_index.json"
                    )
                else:
                    from huggingface_hub import hf_hub_download

                    # Try to download model_index.json
                    config_path = hf_hub_download(
                        repo_id=huggingface_repo.repo_id,
                        filename="model_index.json",
                        revision=huggingface_repo.revision,
                    )

                # Load the config
                with open(config_path) as f:
                    config = json.load(f)

                self._cached_diffusers_configs[huggingface_repo] = config
            except Exception as e:
                # If model_index.json doesn't exist, this is not a diffusion pipeline
                logger.debug(
                    f"No diffusers config found for {huggingface_repo.repo_id}: {e}"
                )
                self._cached_diffusers_configs[huggingface_repo] = None

        return self._cached_diffusers_configs[huggingface_repo]

    def get_active_tokenizer(
        self, huggingface_repo: HuggingFaceRepo
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Retrieves or creates a cached Hugging Face AutoTokenizer for the given model.

        Maintains a cache of Hugging Face tokenizers to avoid
        reloading them unnecessarily which incurs a Hugging Face Hub API call.
        If a tokenizer for the given model hasn't been loaded before, it will
        create a new one using AutoTokenizer.from_pretrained() with the model's
        settings.

        Args:
            huggingface_repo: The HuggingFaceRepo containing the model.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The Hugging Face tokenizer for the model.
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

    def _resolve_architecture(
        self, name: str, task: PipelineTask | None = None
    ) -> SupportedArchitecture | None:
        """Look up an architecture by name, optionally disambiguating by task.

        When multiple architectures share the same name (e.g., a text generation
        model and a TTS model both using LlamaForCausalLM), the task parameter
        allows selecting the correct one.

        Args:
            name: The architecture name to look up.
            task: Optional task to disambiguate when multiple architectures
                share the same name.

        Returns:
            The matching SupportedArchitecture, or None if not found.
        """
        if task is not None:
            task_key = (name, task)
            if task_key in self._architectures_by_task:
                return self._architectures_by_task[task_key]
        return self.architectures.get(name)

    def retrieve_tokenizer(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> PipelineTokenizer[Any, Any, Any]:
        """Retrieves a tokenizer for the given pipeline configuration.

        Args:
            pipeline_config: Configuration for the pipeline
            override_architecture: Optional architecture override string
            task: Optional pipeline task to disambiguate when multiple
                architectures share the same name but serve different tasks.

        Returns:
            PipelineTokenizer: The configured tokenizer

        Raises:
            ValueError: If no architecture is found
        """
        # MAX pipeline
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                huggingface_repo=pipeline_config.model.huggingface_model_repo,
                use_legacy_module=pipeline_config.use_legacy_module,
                task=task,
            )

        if arch is None:
            raise ValueError(
                f"No architecture found for {pipeline_config.model.huggingface_model_repo.repo_id}"
            )

        # Calculate Max Length
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required to initialize tokenizer for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        # Use ArchConfigWithKVCache if available for max_seq_len
        if issubclass(arch.config, ArchConfigWithKVCache):
            arch_config = arch.config.initialize(pipeline_config)
            max_length = arch_config.get_max_seq_len()
        else:
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
                pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=pipeline_config.retrieve_chat_template(),
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                chat_template=pipeline_config.retrieve_chat_template(),
            )

        return tokenizer

    def retrieve_factory(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer[Any, Any, Any], Callable[[], PipelineTypes]]:
        """Retrieves the tokenizer and a factory that creates the pipeline instance."""
        tokenizer: PipelineTokenizer[Any, Any, Any]
        pipeline_factory: Callable[[], PipelineTypes]

        pipeline_class = get_pipeline_for_task(task, pipeline_config)

        # MAX pipeline
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                huggingface_repo=pipeline_config.model.huggingface_model_repo,
                use_legacy_module=pipeline_config.use_legacy_module,
                task=task,
            )

        # Architecture should not be None here, as the engine is MAX.
        if arch is None:
            raise ValueError(
                f"No architecture found for {pipeline_config.model.huggingface_model_repo.repo_id}"
            )

        # For pixel generation (diffusion models), we don't need HuggingFace transformers config
        if task == PipelineTask.PIXEL_GENERATION:
            # Pixel generation pipelines use a different tokenizer with subfolder parameters
            # Check if there's a secondary tokenizer (tokenizer_2) in the diffusers config
            diffusers_config = pipeline_config.model.diffusers_config
            has_tokenizer_2 = False
            if diffusers_config and "components" in diffusers_config:
                has_tokenizer_2 = (
                    "tokenizer_2" in diffusers_config["components"]
                )

            # Standard max_length for CLIP tokenizer (primary)
            # and T5 tokenizer (secondary, if present)
            tokenizer_kwargs = {
                "model_path": pipeline_config.model.model_path,
                "pipeline_config": pipeline_config,
                "subfolder": "tokenizer",
                "max_length": 77,  # Standard for CLIP
                "revision": pipeline_config.model.huggingface_model_revision,
                "trust_remote_code": pipeline_config.model.trust_remote_code,
            }

            if has_tokenizer_2:
                tokenizer_kwargs["subfolder_2"] = "tokenizer_2"
                tokenizer_kwargs["secondary_max_length"] = (
                    512  # Standard for T5
                )

            tokenizer = arch.tokenizer(**tokenizer_kwargs)

            # Pixel generation pipeline only needs pipeline_config and pipeline_model
            pixel_factory_kwargs: dict[str, Any] = {
                "pipeline_config": pipeline_config,
                "pipeline_model": arch.pipeline_model,
            }

            pipeline_factory = cast(
                Callable[[], PipelineTypes],
                functools.partial(  # type: ignore
                    pipeline_class, **pixel_factory_kwargs
                ),
            )

            # Cast tokenizer for return (pixel generation tokenizer doesn't have eos)
            typed_tokenizer = cast(
                PipelineTokenizer[Any, Any, Any],
                tokenizer,
            )

            return typed_tokenizer, pipeline_factory

        # Load HuggingFace Config for text generation and other tasks
        huggingface_config = pipeline_config.model.huggingface_config

        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required to initialize pipeline for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        arch_config = arch.config.initialize(pipeline_config)
        max_length = arch_config.get_max_seq_len()

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
                pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
                enable_llama_whitespace_fix=True,
                chat_template=pipeline_config.retrieve_chat_template(),
                context_validators=arch.context_validators,
            )
        else:
            tokenizer = arch.tokenizer(
                model_path=pipeline_config.model.model_path,
                pipeline_config=pipeline_config,
                revision=pipeline_config.model.huggingface_model_revision,
                max_length=max_length,
                trust_remote_code=pipeline_config.model.trust_remote_code,
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

        # For speculative decoding, retrieve draft model's architecture
        factory_kwargs: dict[str, Any] = {
            "pipeline_config": pipeline_config,
            "pipeline_model": arch.pipeline_model,
            "eos_token_id": tokenizer.eos,
            "weight_adapters": arch.weight_adapters,
            "tokenizer": typed_tokenizer,
        }

        # If using speculative decoding, add draft model-specific parameters
        if pipeline_config.draft_model is not None:
            draft_arch = self.retrieve_architecture(
                huggingface_repo=pipeline_config.draft_model.huggingface_weight_repo,
                use_legacy_module=pipeline_config.use_legacy_module,
                task=task,
            )
            if draft_arch is None:
                raise ValueError(
                    f"MAX-Optimized architecture not found for draft model "
                    f"'{pipeline_config.draft_model.model_path}'"
                )
            factory_kwargs["draft_pipeline_model"] = draft_arch.pipeline_model
            factory_kwargs["draft_weight_adapters"] = draft_arch.weight_adapters

        pipeline_factory = cast(
            Callable[[], PipelineTypes],
            functools.partial(  # type: ignore
                pipeline_class, **factory_kwargs
            ),
        )

        if tokenizer.eos is None:
            raise ValueError(
                "tokenizer.eos value is None, tokenizer configuration is incomplete."
            )

        return tokenizer, pipeline_factory

    def retrieve_context_type(
        self,
        pipeline_config: PipelineConfig,
        override_architecture: str | None = None,
        task: PipelineTask | None = None,
    ) -> type[TextGenerationContext] | type[EmbeddingsContext]:
        """Retrieve the context class type associated with the architecture for the given pipeline configuration.

        The context type defines how the pipeline manages request state and inputs during
        model execution. Different architectures may use different context implementations
        that adhere to either the TextGenerationContext or EmbeddingsContext protocol.

        Args:
            pipeline_config: The configuration for the pipeline.
            override_architecture: Optional architecture name to use instead of looking up
                based on the model repository. This is useful for cases like audio generation
                where the pipeline uses a different architecture (e.g., audio decoder) than
                the underlying model repository.
            task: Optional pipeline task to disambiguate when multiple architectures share
                the same name but serve different tasks.

        Returns:
            The context class type associated with the architecture, which implements
            either the TextGenerationContext or EmbeddingsContext protocol.

        Raises:
            ValueError: If no supported architecture is found for the given model repository
                or override architecture name.
        """
        if override_architecture:
            arch = self._resolve_architecture(override_architecture, task)
        else:
            arch = self.retrieve_architecture(
                huggingface_repo=pipeline_config.model.huggingface_model_repo,
                use_legacy_module=pipeline_config.use_legacy_module,
                task=task,
            )

        if arch:
            return arch.context_type

        raise ValueError(
            f"No architecture found for {pipeline_config.model.huggingface_model_repo.repo_id}"
        )

    def retrieve_pipeline_task(
        self, pipeline_config: PipelineConfig
    ) -> PipelineTask:
        """Retrieves the pipeline task for the given pipeline configuration.

        Args:
            pipeline_config: The configuration for the pipeline.

        Returns:
            The task associated with the architecture.

        Raises:
            ValueError: If no supported architecture is found for the given model repository.
        """
        if arch := self.retrieve_architecture(
            huggingface_repo=pipeline_config.model.huggingface_model_repo,
            use_legacy_module=pipeline_config.use_legacy_module,
        ):
            return arch.task

        raise ValueError(
            f"MAX Optimized architecture not supported for {pipeline_config.model.huggingface_model_repo.repo_id}"
        )

    def retrieve(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        override_architecture: str | None = None,
    ) -> tuple[PipelineTokenizer[Any, Any, Any], PipelineTypes]:
        """Retrieves the tokenizer and an instantiated pipeline for the config."""
        tokenizer, pipeline_factory = self.retrieve_factory(
            pipeline_config, task, override_architecture
        )
        return tokenizer, pipeline_factory()

    def reset(self) -> None:
        """Clears all registered architectures (mainly for tests)."""
        self.architectures.clear()
        self._architectures_by_task.clear()


PIPELINE_REGISTRY = PipelineRegistry([])
