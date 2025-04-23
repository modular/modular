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
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

import torch
from max.driver import Device, load_devices
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import (
    EmbeddingsGenerator,
    PipelineTask,
    PipelineTokenizer,
    TokenGenerator,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from .config import PipelineConfig
from .config_enums import (
    PipelineEngine,
    RopeType,
    SupportedEncoding,
)
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_pipeline import HFEmbeddingsPipeline, HFTextGenerationPipeline
from .hf_utils import HuggingFaceRepo
from .pipeline import PipelineModel, TextGenerationPipeline
from .speculative_decoding import SpeculativeDecodingTextGenerationPipeline
from .tokenizer import TextAndVisionTokenizer, TextTokenizer

logger = logging.getLogger("max.pipelines")


def get_pipeline_for_task(
    task: PipelineTask, pipeline_config: PipelineConfig
) -> (
    type[TextGenerationPipeline]
    | type[EmbeddingsPipeline]
    | type[SpeculativeDecodingTextGenerationPipeline]
):
    if task == PipelineTask.TEXT_GENERATION:
        if pipeline_config.draft_model_config is not None:
            return SpeculativeDecodingTextGenerationPipeline
        else:
            return TextGenerationPipeline
    elif task == PipelineTask.EMBEDDINGS_GENERATION:
        return EmbeddingsPipeline
    else:
        msg = f"PipelineTask ({task}) does not have supported Pipeline"
        raise ValueError(msg)


_HF_PIPELINE_TASK_MAP: dict[
    PipelineTask, type[HFTextGenerationPipeline] | type[HFEmbeddingsPipeline]
] = {
    PipelineTask.TEXT_GENERATION: HFTextGenerationPipeline,
    PipelineTask.EMBEDDINGS_GENERATION: HFEmbeddingsPipeline,
}


class SupportedArchitecture:
    def __init__(
        self,
        name: str,
        example_repo_ids: list[str],
        default_encoding: SupportedEncoding,
        supported_encodings: dict[SupportedEncoding, list[KVCacheStrategy]],
        pipeline_model: type[PipelineModel],
        task: PipelineTask,
        tokenizer: type[Union[TextTokenizer, TextAndVisionTokenizer]],
        default_weights_format: WeightsFormat,
        multi_gpu_supported: bool = False,
        rope_type: RopeType = RopeType.none,
        weight_adapters: dict[WeightsFormat, WeightsAdapter] | None = None,
    ):
        """Initializes a model architecture supported by MAX pipelines.

        New architectures should be registered into the :obj:`PipelineRegistry`.

        args:
            name: Architecture name.
            example_repo_ids: Hugging Face `repo_id` which runs this architecture.
            default_encoding: Default encoding for the model.
            supported_encodings: Alternate encodings supported.
            pipeline_model: :obj:`PipelineModel` class that defines the model graph
                and execution.
            task: Which pipeline task should the model run with.
            tokenizer: Tokenizer used to preprocess model inputs.
            default_weights_format: The weights format used in `pipeline_model`.
            weight_converters: A dictionary of weight loaders to use if the
                input checkpoint has a different format than the default.
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

    def __eq__(self, other) -> bool:
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


class PipelineRegistry:
    def __init__(self, architectures: list[SupportedArchitecture]):
        self.architectures = {arch.name: arch for arch in architectures}
        self._cached_huggingface_configs: dict[HuggingFaceRepo, AutoConfig] = {}
        self._cached_huggingface_tokenizers: dict[
            HuggingFaceRepo, PreTrainedTokenizer | PreTrainedTokenizerFast
        ] = {}

    def register(self, architecture: SupportedArchitecture):
        """Add new architecture to registry."""
        if architecture.name in self.architectures:
            msg = f"Refusing to override existing architecture for '{architecture.name}'"
            raise ValueError(msg)

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
        message = f"""

        Loading {tokenizer_type.__name__} and {pipeline_name}({pipeline_model}) {factory_str} for:
            engine:                 {pipeline_config.engine}
            architecture:           {architecture_id if architecture_id else "UNKNOWN"}
            devices:                {devices_str}
            model_path:             {pipeline_config.model_config.model_path}{weights_repo_str}
            huggingface_revision:   {pipeline_config.model_config.huggingface_model_revision}
            quantization_encoding:  {pipeline_config.model_config.quantization_encoding}
            cache_strategy:         {pipeline_config.model_config.kv_cache_config.cache_strategy}
            weight_path:            [
        {weight_path}
                                    ]
        """

        return message

    def _set_hf_pipeline_defaults(
        self, pipeline_config: PipelineConfig
    ) -> PipelineConfig:
        if pipeline_config.max_batch_size is None:
            pipeline_config.max_batch_size = 1
        # HF pipelines always use custom continuous cache
        pipeline_config.model_config.kv_cache_config.cache_strategy = (
            KVCacheStrategy.CONTINUOUS
        )
        return pipeline_config

    def retrieve_factory(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
    ) -> tuple[
        PipelineTokenizer,
        Callable[[], TokenGenerator | EmbeddingsGenerator],
    ]:
        tokenizer: PipelineTokenizer
        pipeline_factory: Callable[[], TokenGenerator | EmbeddingsGenerator]

        if pipeline_config.engine == PipelineEngine.MAX:
            pipeline_class = get_pipeline_for_task(task, pipeline_config)

            # MAX pipeline
            arch = self.retrieve_architecture(
                huggingface_repo=pipeline_config.model_config.huggingface_model_repo,
            )

            # Load HuggingFace Config
            huggingface_config = pipeline_config.model_config.huggingface_config

            # Architecture should not be None here, as the engine is MAX.
            assert arch is not None
            devices = load_devices(pipeline_config.model_config.device_specs)
            logger.info(
                self._load_logging_message(
                    pipeline_config=pipeline_config,
                    tokenizer_type=arch.tokenizer,
                    pipeline_model=arch.pipeline_model.__name__,
                    pipeline_name=pipeline_class.__name__,
                    architecture_id=arch.name,
                    factory=True,
                    devices=devices,
                )
            )

            max_length = arch.pipeline_model.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
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
                    pipeline_config.model_config.model_path,
                    revision=pipeline_config.model_config.huggingface_model_revision,
                    max_length=max_length,
                    max_new_tokens=pipeline_config.max_new_tokens,
                    trust_remote_code=pipeline_config.model_config.trust_remote_code,
                )

            pipeline_factory = functools.partial(
                pipeline_class,
                pipeline_config=pipeline_config,
                pipeline_model=arch.pipeline_model,
                eos_token_id=tokenizer.eos,
                weight_adapters=arch.weight_adapters,
            )
        else:
            pipeline_config = self._set_hf_pipeline_defaults(pipeline_config)
            hf_pipeline_class = _HF_PIPELINE_TASK_MAP[task]

            torch_device_type = str(
                pipeline_config.model_config.device_specs[0].device_type
            )
            if (
                pipeline_config.model_config.device_specs[0].device_type
                == "gpu"
            ):
                torch_device_type = "cuda"
                torch.multiprocessing.set_start_method("spawn", force=True)

            # Generalized pipeline
            tokenizer = TextTokenizer(
                pipeline_config.model_config.model_path,
                revision=pipeline_config.model_config.huggingface_model_revision,
                max_length=pipeline_config.max_length,
                max_new_tokens=pipeline_config.max_new_tokens,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                enable_llama_whitespace_fix=True,
            )
            logger.info(
                self._load_logging_message(
                    pipeline_config=pipeline_config,
                    tokenizer_type=TextTokenizer,
                    pipeline_model="",
                    pipeline_name=hf_pipeline_class.__name__,
                    factory=True,
                    devices=load_devices(
                        pipeline_config.model_config.device_specs
                    ),
                )
            )
            pipeline_factory = functools.partial(
                hf_pipeline_class,
                pipeline_config=pipeline_config,
                torch_device_type=torch_device_type,
            )

        if tokenizer.eos is None:
            msg = "tokenizer.eos value is None, tokenizer configuration is incomplete."
            raise ValueError(msg)

        return tokenizer, pipeline_factory

    def retrieve(
        self,
        pipeline_config: PipelineConfig,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
    ) -> tuple[PipelineTokenizer, TokenGenerator | EmbeddingsGenerator]:
        tokenizer, pipeline_factory = self.retrieve_factory(
            pipeline_config, task
        )
        return tokenizer, pipeline_factory()

    def reset(self) -> None:
        self.architectures.clear()


PIPELINE_REGISTRY = PipelineRegistry([])
