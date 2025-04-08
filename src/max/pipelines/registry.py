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
from io import StringIO
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

import torch
from max.driver import Device, load_devices
from max.dtype import DType
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

if TYPE_CHECKING:
    from .config import PipelineConfig
from .config_enums import (
    PipelineEngine,
    RopeType,
    SupportedEncoding,
)
from .core import (
    EmbeddingsGenerator,
    PipelineTask,
    PipelineTokenizer,
    TokenGenerator,
)
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_pipeline import HFEmbeddingsPipeline, HFTextGenerationPipeline
from .hf_utils import get_architectures_from_huggingface_repo
from .kv_cache import KVCacheStrategy
from .max_config import HuggingFaceRepo, KVCacheConfig, MAXModelConfig
from .pipeline import KVCacheMixin, PipelineModel, TextGenerationPipeline
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
        if pipeline_config.draft_model is not None:
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

    def register(self, architecture: SupportedArchitecture):
        """Add new architecture to registry."""
        if architecture.name in self.architectures:
            msg = f"Refusing to override existing architecture for '{architecture.name}'"
            raise ValueError(msg)

        self.architectures[architecture.name] = architecture

    def retrieve_architecture(
        self,
        model_path: str,
        trust_remote_code: bool,
        huggingface_revision: str,
    ) -> Optional[SupportedArchitecture]:
        # Retrieve model architecture names
        architecture_names = get_architectures_from_huggingface_repo(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
            huggingface_revision=huggingface_revision,
        )

        if not architecture_names:
            logger.debug(
                "architectures not listed in HuggingFace config, cannot be matched against MAX Registry"
            )
            return None

        for architecture_name in architecture_names:
            if architecture_name in self.architectures:
                return self.architectures[architecture_name]

        logger.debug(
            f"optimized architecture not available for {model_path} in MAX REGISTRY"
        )

        return None

    def _get_active_huggingface_config(
        self, model_config: MAXModelConfig
    ) -> AutoConfig:
        """Retrieves or creates a cached HuggingFace AutoConfig for the given
        model configuration.

        This method maintains a cache of HuggingFace configurations to avoid
        reloading them unnecessarily which incurs a huggingface hub API call.
        If a config for the given model hasn't been loaded before, it will
        create a new one using AutoConfig.from_pretrained() with the model's
        settings.

        Args:
            model_config: The MAX model configuration containing model path and
            loading settings.

        Returns:
            AutoConfig: The HuggingFace configuration object for the model.
        """
        if (
            model_config.huggingface_weights_repo
            not in self._cached_huggingface_configs
        ):
            self._cached_huggingface_configs[
                model_config.huggingface_weights_repo
            ] = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                revision=model_config.huggingface_revision,
            )

        return self._cached_huggingface_configs[
            model_config.huggingface_weights_repo
        ]

    def _estimate_memory_footprint(
        self,
        pipeline_config: PipelineConfig,
        arch: SupportedArchitecture,
        devices: list[Device],
    ):
        model_cls = arch.pipeline_model
        model_config = pipeline_config.model_config

        huggingface_config = self._get_active_huggingface_config(model_config)

        try:
            free_memory = int(sum(d.stats["free_memory"] for d in devices))
        except Exception as e:
            logger.warning(
                "Unable to estimate memory footprint of model, can't query device stats: "
                + str(e)
            )
            if not pipeline_config.max_batch_size:
                pipeline_config.max_batch_size = 1
            if not pipeline_config.max_length:
                pipeline_config.max_length = model_cls.calculate_max_seq_len(
                    pipeline_config,
                    huggingface_config=huggingface_config,
                )
            return

        model_weights_size = model_cls.estimate_weights_size(pipeline_config)

        if model_weights_size > free_memory:
            raise RuntimeError(
                f"Model size exceeds available memory ({to_human_readable_bytes(model_weights_size)} > {to_human_readable_bytes(free_memory)}). "
                "Try running a smaller model, using a smaller precision, or using a device with more memory."
            )

        total_size = model_weights_size
        available_kv_cache_memory = int(
            free_memory * model_config.kv_cache_config.device_memory_utilization
            - model_weights_size
        )
        available_kv_cache_memory = max(0, available_kv_cache_memory)

        user_provided_max_length = pipeline_config.max_length is not None
        user_provided_max_batch_size = (
            pipeline_config.max_batch_size is not None
        )
        if not user_provided_max_length:
            pipeline_config.max_length = model_cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            )

        if not model_config.quantization_encoding:
            msg = "quantization_encoding must be provided in pipeline_config"
            raise ValueError(msg)

        if not user_provided_max_batch_size:
            pipeline_config.max_batch_size = self._infer_optimal_batch_size(
                pipeline_config,
                model_cls,
                available_kv_cache_memory,
                huggingface_config=huggingface_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=model_config.quantization_encoding.cache_dtype,
            )

        actual_kv_cache_size = self._calculate_kv_cache_size(
            model_cls,
            pipeline_config,
            available_kv_cache_memory,
            huggingface_config,
            devices=devices,
            kv_cache_config=model_config.kv_cache_config,
            cache_dtype=model_config.quantization_encoding.cache_dtype,
        )

        model_config.kv_cache_config._available_cache_memory = (
            actual_kv_cache_size
        )

        total_size += actual_kv_cache_size
        # If the model is too large to fit in memory, and the user did not
        # specify a max_length, try to infer a value that would fit.
        if total_size > free_memory and not user_provided_max_length:
            original_max_length = pipeline_config.max_length
            (
                found_valid_max_length,
                inferred_max_length,
                _,
            ) = self._find_valid_max_length(
                pipeline_config,
                model_cls,
                available_kv_cache_memory,
                user_provided_max_batch_size,
                huggingface_config=huggingface_config,
                devices=devices,
            )

            if found_valid_max_length:
                logger.warning(
                    f"Truncated model's default max_length from {original_max_length} to {inferred_max_length} to fit in memory."
                )
                pipeline_config.max_length = inferred_max_length
            else:
                pipeline_config.max_length = 1

            actual_kv_cache_size = self._calculate_kv_cache_size(
                model_cls,
                pipeline_config,
                available_kv_cache_memory,
                huggingface_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=model_config.quantization_encoding.cache_dtype,
            )
            total_size = model_weights_size + actual_kv_cache_size

        if free_memory:
            free_memory_str = f" / {to_human_readable_bytes(free_memory)} free"

        weights_str = ""
        if model_weights_size:
            weights_str = f"\n\t    Weights:                {to_human_readable_bytes(model_weights_size)}"

        if not user_provided_max_length:
            max_length_str = f"Auto-inferred max sequence length: {pipeline_config.max_length}"
        else:
            max_length_str = (
                f"Current max sequence length: {pipeline_config.max_length}"
            )

        if not user_provided_max_batch_size:
            max_batch_size_str = f"Auto-inferred max batch size: {pipeline_config.max_batch_size}"
        else:
            max_batch_size_str = (
                f"Current max batch size: {pipeline_config.max_batch_size}"
            )

        logging_str = (
            "\n"
            f"\n\tEstimated memory consumption:"
            f"{weights_str}"
            f"\n\t    KVCache allocation:     {to_human_readable_bytes(actual_kv_cache_size)}"
            f"\n\t    Total estimated:        {to_human_readable_bytes(model_weights_size + actual_kv_cache_size)} used{free_memory_str}"
            f"\n\t{max_length_str}"
            f"\n\t{max_batch_size_str}\n"
        )
        logger.info(logging_str)
        vram_usage_limit_scale = 0.95

        if isinstance(free_memory, (int, float)):
            if total_size > free_memory:
                self._raise_oom_error(
                    pipeline_config,
                    user_provided_max_length,
                    user_provided_max_batch_size,
                    model_cls,
                    total_size,
                    free_memory,
                    available_kv_cache_memory,
                    model_weights_size,
                    huggingface_config,
                    devices=devices,
                )

            elif total_size > vram_usage_limit_scale * free_memory:
                logger.warning(
                    "Estimated model and kv cache memory use nears available memory. You may experience errors."
                )

    def _raise_oom_error(
        self,
        pipeline_config: PipelineConfig,
        user_provided_max_length: bool,
        user_provided_max_batch_size: bool,
        model_cls: type[PipelineModel],
        total_size: int,
        original_free_memory: int,
        available_kv_cache_memory: int,
        weights_size: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
    ) -> None:
        """If we've determined the current configuration won't fit in device memory,
        this method provides a friendly error message suggesting a viable configuration.

        The approach is to:
        1. Binary search max_length until we find a setting that works
        2. If user provided max_batch_size, binary search that too
        3. Generate appropriate suggestions based on this truth table:

                                                            max_length
                                         +----------------------+--------------------------+
                                         | set by user          | set to default           |
                        +----------------+======================+==========================+
                        | set by user    ║ Recommend both       | Recommend max_batch_size |
        max_batch_size  +----------------+----------------------+--------------------------+
                        | set to default ║ Recommend max_length | Recommend both           |
                        +----------------+----------------------+--------------------------+
        """
        original_max_length = cast(int, pipeline_config.max_length)
        original_max_batch_size = cast(int, pipeline_config.max_batch_size)

        # Find valid configurations through binary search
        (
            found_valid_max_length,
            inferred_max_length,
            inferred_max_length_compatible_batch_size,
        ) = self._find_valid_max_length(
            pipeline_config,
            model_cls,
            available_kv_cache_memory,
            user_provided_max_batch_size,
            huggingface_config,
            devices=devices,
        )

        pipeline_config.max_batch_size = original_max_batch_size

        found_valid_max_batch_size, inferred_max_batch_size = (
            self._find_valid_batch_size(
                pipeline_config,
                model_cls,
                available_kv_cache_memory,
                original_max_length,
                user_provided_max_batch_size,
                huggingface_config,
                devices=devices,
            )
        )

        # Generate error message with suggestions
        error_msg = self._generate_oom_error_message(
            total_size=total_size,
            original_free_memory=original_free_memory,
            user_provided_max_length=user_provided_max_length,
            user_provided_max_batch_size=user_provided_max_batch_size,
            found_valid_max_length=found_valid_max_length,
            found_valid_max_batch_size=found_valid_max_batch_size,
            inferred_max_length=inferred_max_length,
            inferred_max_batch_size=inferred_max_batch_size,
            inferred_max_length_compatible_batch_size=inferred_max_length_compatible_batch_size,
            original_max_length=original_max_length,
        )

        raise RuntimeError(error_msg)

    def _find_valid_max_length(
        self,
        pipeline_config: PipelineConfig,
        model_cls: type[PipelineModel],
        available_kv_cache_memory: int,
        user_provided_max_batch_size: bool,
        huggingface_config: AutoConfig,
        devices: list[Device],
    ) -> tuple[bool, int, int]:
        """Binary search to find a valid max_length configuration.

        Returns:
            Tuple containing:
            - found_valid_max_length: Whether a valid max_length was found
            - inferred_max_length: The suggested max_length value
            - inferred_max_length_compatible_batch_size: Compatible batch size for the max_length
        """
        assert pipeline_config.max_length is not None
        assert pipeline_config.max_batch_size is not None

        found_valid_max_length = False
        lower = 1
        upper = pipeline_config.max_length
        inferred_max_length = upper

        model_config = pipeline_config.model_config
        if not model_config.quantization_encoding:
            msg = "quantization_encoding must be provided in pipeline_config"
            raise ValueError(msg)

        while not found_valid_max_length:
            inferred_max_length = (lower + upper) // 2
            pipeline_config.max_length = inferred_max_length

            if not user_provided_max_batch_size:
                pipeline_config.max_batch_size = self._infer_optimal_batch_size(
                    pipeline_config,
                    model_cls,
                    available_kv_cache_memory,
                    huggingface_config,
                    devices=devices,
                    kv_cache_config=model_config.kv_cache_config,
                    cache_dtype=model_config.quantization_encoding.cache_dtype,
                )

            kv_cache_size = self._calculate_kv_cache_size(
                model_cls,
                pipeline_config,
                available_kv_cache_memory,
                huggingface_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=model_config.quantization_encoding.cache_dtype,
            )

            if lower > upper:
                break
            elif upper - lower <= 1:
                if kv_cache_size <= available_kv_cache_memory:
                    found_valid_max_length = True
                break

            if kv_cache_size > available_kv_cache_memory:
                upper = inferred_max_length - 1
            else:
                lower = inferred_max_length
        return (
            found_valid_max_length,
            inferred_max_length,
            pipeline_config.max_batch_size,
        )

    def _find_valid_batch_size(
        self,
        pipeline_config: PipelineConfig,
        model_cls: type[PipelineModel],
        available_kv_cache_memory: int,
        original_max_length: int,
        user_provided_max_batch_size: bool,
        huggingface_config: AutoConfig,
        devices: list[Device],
    ) -> tuple[bool, int]:
        """Binary search to find a valid batch size configuration.

        Returns:
            Tuple containing:
            - found_valid_max_batch_size: Whether a valid batch size was found
            - inferred_max_batch_size: The suggested batch size value.
                If the user did not provide a batch size, this will be -1.
        """
        if not user_provided_max_batch_size:
            return False, -1

        found_valid_max_batch_size = False
        pipeline_config.max_length = original_max_length
        inferred_max_batch_size = cast(int, pipeline_config.max_batch_size)
        lower = 1
        upper = cast(int, pipeline_config.max_batch_size)
        model_config = pipeline_config.model_config

        while not found_valid_max_batch_size:
            inferred_max_batch_size = (lower + upper) // 2
            pipeline_config.max_batch_size = inferred_max_batch_size

            if not model_config.quantization_encoding:
                msg = (
                    "quantization_encoding must be provided in pipeline_config"
                )
                raise ValueError(msg)

            kv_cache_size = self._calculate_kv_cache_size(
                model_cls,
                pipeline_config,
                available_kv_cache_memory,
                huggingface_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=model_config.quantization_encoding.cache_dtype,
            )

            if lower > upper:
                break
            elif upper - lower <= 1:
                if kv_cache_size <= available_kv_cache_memory:
                    found_valid_max_batch_size = True
                break

            if kv_cache_size > available_kv_cache_memory:
                upper = inferred_max_batch_size - 1
            else:
                lower = inferred_max_batch_size

        return found_valid_max_batch_size, inferred_max_batch_size

    def _calculate_kv_cache_size(
        self,
        model_cls: type[PipelineModel],
        pipeline_config: PipelineConfig,
        available_kv_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Calculate the KV cache size for the current configuration."""
        if issubclass(model_cls, KVCacheMixin):
            return model_cls.estimate_kv_cache_size(
                pipeline_config=pipeline_config,
                available_cache_memory=available_kv_cache_memory,
                devices=devices,
                huggingface_config=huggingface_config,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            )
        return 0

    def _generate_oom_error_message(
        self,
        total_size: int,
        original_free_memory: int,
        user_provided_max_length: bool,
        user_provided_max_batch_size: bool,
        found_valid_max_length: bool,
        found_valid_max_batch_size: bool,
        inferred_max_length: int,
        inferred_max_batch_size: int,
        inferred_max_length_compatible_batch_size: int,
        original_max_length: int,
    ) -> str:
        """Generate an appropriate error message based on the configuration state."""
        free_memory_str = (
            f" / {to_human_readable_bytes(original_free_memory)} free"
            if original_free_memory
            else ""
        )

        msg = StringIO()
        msg.write(
            f"Estimated model and kv cache memory use exceeds available memory ({to_human_readable_bytes(total_size)} {free_memory_str}). Try "
        )

        if not found_valid_max_length and not found_valid_max_batch_size:
            msg.write(
                "reducing --max-length or --max-batch-size, finding a smaller model, or using a device with more memory."
            )

        elif user_provided_max_length:
            self._add_user_provided_max_length_suggestions(
                msg,
                user_provided_max_batch_size,
                found_valid_max_length,
                found_valid_max_batch_size,
                inferred_max_length,
                inferred_max_batch_size,
                inferred_max_length_compatible_batch_size,
            )
        else:
            self._add_default_max_length_suggestions(
                msg,
                user_provided_max_batch_size,
                found_valid_max_length,
                found_valid_max_batch_size,
                inferred_max_length,
                inferred_max_batch_size,
                inferred_max_length_compatible_batch_size,
                original_max_length,
            )

        msg.write(".")
        return msg.getvalue()

    def _add_user_provided_max_length_suggestions(
        self,
        msg: StringIO,
        user_provided_max_batch_size: bool,
        found_valid_max_length: bool,
        found_valid_max_batch_size: bool,
        inferred_max_length: int,
        inferred_max_batch_size: int,
        inferred_max_length_compatible_batch_size: int,
    ) -> None:
        """Add error message suggestions when user provided max_length.

        This handles the top row of the truth table from the _raise_oom_error docstring.

        Args:
            msg: StringIO buffer to write message to
            user_provided_max_batch_size: Whether user provided batch size
            found_valid_max_length: Whether valid max_length was found
            found_valid_max_batch_size: Whether valid batch size was found
            inferred_max_length: Suggested max_length value
            inferred_max_batch_size: Suggested batch size value
            inferred_max_length_compatible_batch_size: Compatible batch size for max_length
        """
        if not user_provided_max_batch_size:
            if found_valid_max_length:
                msg.write(
                    f"reducing --max-length to {inferred_max_length} "
                    f"(supports batch size of {inferred_max_length_compatible_batch_size})"
                )
            else:
                msg.write("reducing --max-length or --max-batch-size")
        else:
            if found_valid_max_length:
                msg.write(
                    f"reducing --max-length to {inferred_max_length} and "
                    f"--max-batch-size to {inferred_max_length_compatible_batch_size})"
                )

            if found_valid_max_batch_size:
                if found_valid_max_length:
                    msg.write(" or ")
                msg.write(
                    f"reducing --max-batch-size to {inferred_max_batch_size}"
                )

    def _add_default_max_length_suggestions(
        self,
        msg: StringIO,
        user_provided_max_batch_size: bool,
        found_valid_max_length: bool,
        found_valid_max_batch_size: bool,
        inferred_max_length: int,
        inferred_max_batch_size: int,
        inferred_max_length_compatible_batch_size: int,
        original_max_length: int,
    ) -> None:
        """Add error message suggestions when max_length was set to default.

        This handles the bottom row of the truth table from the _raise_oom_error docstring.

        Args:
            msg: StringIO buffer to write message to
            user_provided_max_batch_size: Whether user provided batch size
            found_valid_max_length: Whether valid max_length was found
            found_valid_max_batch_size: Whether valid batch size was found
            inferred_max_length: Suggested max_length value
            inferred_max_batch_size: Suggested batch size value
            inferred_max_length_compatible_batch_size: Compatible batch size for max_length
            original_max_length: Original max_length value before modifications
        """
        if not user_provided_max_batch_size:
            if found_valid_max_length:
                msg.write(
                    f"setting --max-length to {inferred_max_length} and "
                    f"--max-batch-size to {inferred_max_length_compatible_batch_size})"
                )

            if found_valid_max_batch_size:
                if found_valid_max_length:
                    msg.write(" or ")
                msg.write(
                    f"setting --max-batch-size to {inferred_max_batch_size}"
                )

        else:
            if found_valid_max_batch_size:
                msg.write(
                    f"reducing --max-batch-size to {inferred_max_batch_size}"
                )
            if found_valid_max_length:
                if found_valid_max_batch_size:
                    msg.write(" or ")
                msg.write(
                    f"setting --max-length to {inferred_max_length} "
                    f"(currently defaulted to {original_max_length})"
                )

    def _infer_optimal_batch_size(
        self,
        pipeline_config: PipelineConfig,
        model_cls: type[PipelineModel],
        available_kv_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        return model_cls.infer_optimal_batch_size(
            pipeline_config,
            available_kv_cache_memory,
            huggingface_config=huggingface_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

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
            huggingface_revision:   {pipeline_config.model_config.huggingface_revision}
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
                model_path=pipeline_config.model_config.model_path,
                trust_remote_code=pipeline_config.model_config.trust_remote_code,
                huggingface_revision=pipeline_config.model_config.huggingface_revision,
            )

            # Load HuggingFace Config
            huggingface_config = self._get_active_huggingface_config(
                pipeline_config.model_config
            )

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
                    revision=pipeline_config.model_config.huggingface_revision,
                    max_length=max_length,
                    max_new_tokens=pipeline_config.max_new_tokens,
                    trust_remote_code=pipeline_config.model_config.trust_remote_code,
                    enable_llama_whitespace_fix=True,
                )
            else:
                tokenizer = arch.tokenizer(
                    pipeline_config.model_config.model_path,
                    revision=pipeline_config.model_config.huggingface_revision,
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
                revision=pipeline_config.model_config.huggingface_revision,
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
