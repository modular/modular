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

import logging
from io import StringIO
from typing import TYPE_CHECKING, Any, cast

from max.driver import Device
from max.dtype import DType
from max.kv_cache import PagedKVCacheManager
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

if TYPE_CHECKING:
    from .config import PipelineConfig


from .interfaces import KVCacheMixin, PipelineModel
from .kv_cache_config import KVCacheConfig
from .model_config import MAXModelConfig

logger = logging.getLogger("max.pipelines")


class MemoryEstimator:
    @classmethod
    def free_memory(cls, devices: list[Device]) -> int:
        """Return the total free memory available across all provided devices."""
        try:
            return int(sum(d.stats["free_memory"] for d in devices))
        except Exception as e:
            logger.warning(
                "Unable to estimate memory footprint of model, can't query device stats: "
                + str(e)
            )
            raise

    @classmethod
    def model_weights_size(cls, model_config: MAXModelConfig) -> int:
        """Return the size of the model weights in bytes."""
        return model_config.weights_size()

    @classmethod
    def activation_memory_size(
        cls,
        pipeline_model: type[PipelineModel[Any]],
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """
        Estimate the activation memory requirement for the model.

        Args:
            pipeline_model: The model class.
            pipeline_config: The pipeline configuration.
            model_config: The model configuration.

        Returns:
            Activation memory size in bytes.
        """
        return pipeline_model.estimate_activation_memory(
            pipeline_config, model_config.huggingface_config
        )

    @classmethod
    def static_memory_size(
        cls,
        pipeline_model: type[PipelineModel[Any]],
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """
        Calculate the static memory usage: model weights plus activations.

        Args:
            pipeline_model: The model class.
            pipeline_config: The pipeline configuration.
            model_config: The model configuration.

        Returns:
            Total static memory usage in bytes.
        """
        return cls.model_weights_size(
            model_config
        ) + cls.activation_memory_size(
            pipeline_model, pipeline_config, model_config
        )

    @classmethod
    def available_kv_cache_memory(
        cls,
        pipeline_model: type[PipelineModel[Any]],
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
        devices: list[Device],
    ) -> int:
        """
        Estimate the available KV cache memory after accounting for model weights and activations.

        Args:
            pipeline_model: The model class.
            pipeline_config: The pipeline configuration.
            model_config: The model configuration.
            devices: The list of devices on which the model will run.

        Returns:
            Available KV cache memory in bytes.
        """
        return int(
            (
                cls.free_memory(devices)
                * model_config.kv_cache_config.device_memory_utilization
            )
            - cls.static_memory_size(
                pipeline_model, pipeline_config, model_config
            )
        )

    @classmethod
    def max_supported_sequence_length(
        cls,
        pipeline_model: type[PipelineModel[Any]],
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
        devices: list[Device],
    ) -> int | None:
        """Compute the hard upper bound on tokens for a single request.

        Mirrors the paged KV cache constraint: per replica, a request cannot
        exceed total pages per device times page size.
        """

        # Retrieve needed parameters.
        if not model_config.quantization_encoding:
            raise ValueError(
                "quantization_encoding must be provided in model_config"
            )

        # Ensure pipeline_model implements KVCacheMixin
        if not issubclass(pipeline_model, KVCacheMixin):
            return None

        kv_cache_model = cast(type[KVCacheMixin], pipeline_model)

        params = kv_cache_model.get_kv_params(
            huggingface_config=model_config.huggingface_config,
            n_devices=len(devices),
            kv_cache_config=model_config.kv_cache_config,
            cache_dtype=model_config.quantization_encoding.cache_dtype,
        )
        num_layers = kv_cache_model.get_num_layers(
            huggingface_config=model_config.huggingface_config
        )

        # Available cache memory
        memory_available = cls.available_kv_cache_memory(
            pipeline_model, pipeline_config, model_config, devices
        )

        return PagedKVCacheManager.max_supported_sequence_length(
            params, num_layers, memory_available
        )

    @classmethod
    def estimate_memory_footprint(
        cls,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[Any]],
        model_config: MAXModelConfig,
        devices: list[Device],
    ) -> None:
        huggingface_config = model_config.huggingface_config
        is_draft_model = (
            pipeline_config.draft_model_config is not None
            and model_config is pipeline_config.draft_model_config
        )

        try:
            free_memory = cls.free_memory(devices)
        except Exception as e:
            if is_draft_model:
                # Early return for draft model - we don't modify the original config
                return
            if not pipeline_config.max_batch_size:
                pipeline_config.max_batch_size = 1
            if not pipeline_config.max_length:
                pipeline_config.max_length = (
                    pipeline_model.calculate_max_seq_len(
                        pipeline_config, huggingface_config=huggingface_config
                    )
                )
            return

        model_weights_size = cls.model_weights_size(model_config)

        # Get activation memory estimate from the model
        activation_memory_size = cls.activation_memory_size(
            pipeline_model, pipeline_config, model_config
        )

        # Total static memory requirement (weights + activations)
        static_memory_size = model_weights_size + activation_memory_size

        if static_memory_size > free_memory:
            error_msg = f"Model size exceeds available memory ({to_human_readable_bytes(static_memory_size)} > {to_human_readable_bytes(free_memory)}). "
            if activation_memory_size > 0:
                error_msg += (
                    f"Model weights: {to_human_readable_bytes(model_weights_size)}, "
                    f"Activation memory: {to_human_readable_bytes(activation_memory_size)}. "
                )
            error_msg += "Try running a smaller model, using a smaller precision, or using a device with more memory."
            raise RuntimeError(error_msg)

        total_size = static_memory_size
        available_kv_cache_memory = int(
            free_memory * model_config.kv_cache_config.device_memory_utilization
            - static_memory_size
        )

        if available_kv_cache_memory <= 0:
            raise RuntimeError(
                f"The model {to_human_readable_bytes(model_weights_size)} and activations "
                f"{to_human_readable_bytes(activation_memory_size)} don't leave room for KV cache. "
                f"Try running a smaller model, using a smaller precision, or using a device with more memory."
            )

        user_provided_max_length = pipeline_config.max_length is not None
        user_provided_max_batch_size = (
            pipeline_config.max_batch_size is not None
        )

        if is_draft_model:
            if not model_config.quantization_encoding:
                raise ValueError(
                    "quantization_encoding must be provided for draft model"
                )

            kv_cache_size = cls._calculate_kv_cache_size(
                pipeline_model,
                pipeline_config,
                available_kv_cache_memory,
                huggingface_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=model_config.quantization_encoding.cache_dtype,
            )

            model_config.kv_cache_config._available_cache_memory = kv_cache_size

            return  # Don't modify pipeline config values

        if not user_provided_max_length:
            pipeline_config.max_length = pipeline_model.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            )

        if not model_config.quantization_encoding:
            raise ValueError(
                "quantization_encoding must be provided in pipeline_config"
            )

        if not user_provided_max_batch_size:
            pipeline_config.max_batch_size = cls._infer_optimal_batch_size(
                pipeline_config,
                pipeline_model,
                available_kv_cache_memory,
                huggingface_config=huggingface_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=model_config.quantization_encoding.cache_dtype,
            )

        actual_kv_cache_size = cls._calculate_kv_cache_size(
            pipeline_model,
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
        if int(total_size) > free_memory and not user_provided_max_length:
            original_max_length = pipeline_config.max_length
            (
                found_valid_max_length,
                inferred_max_length,
                _,
            ) = cls._find_valid_max_length(
                pipeline_config,
                pipeline_model,
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

            actual_kv_cache_size = cls._calculate_kv_cache_size(
                pipeline_model,
                pipeline_config,
                available_kv_cache_memory,
                huggingface_config,
                devices=devices,
                kv_cache_config=model_config.kv_cache_config,
                cache_dtype=model_config.quantization_encoding.cache_dtype,
            )
            total_size = model_weights_size + actual_kv_cache_size

        vram_usage_limit_scale = 0.95

        if isinstance(free_memory, int | float):
            if int(total_size) > int(free_memory):
                cls._raise_oom_error(
                    pipeline_config,
                    user_provided_max_length,
                    user_provided_max_batch_size,
                    pipeline_model,
                    total_size,
                    free_memory,
                    available_kv_cache_memory,
                    model_weights_size,
                    huggingface_config,
                    devices=devices,
                )

            elif int(total_size) > int(vram_usage_limit_scale * free_memory):
                logger.warning(
                    "Estimated model and kv cache memory use nears available memory. You may experience errors."
                )

    @classmethod
    def _find_valid_max_length(
        cls,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[Any]],
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
            raise ValueError(
                "quantization_encoding must be provided in pipeline_config"
            )

        while not found_valid_max_length:
            inferred_max_length = (lower + upper) // 2
            pipeline_config.max_length = inferred_max_length

            if not user_provided_max_batch_size:
                pipeline_config.max_batch_size = cls._infer_optimal_batch_size(
                    pipeline_config,
                    pipeline_model,
                    available_kv_cache_memory,
                    huggingface_config,
                    devices=devices,
                    kv_cache_config=model_config.kv_cache_config,
                    cache_dtype=model_config.quantization_encoding.cache_dtype,
                )

            kv_cache_size = cls._calculate_kv_cache_size(
                pipeline_model,
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

    @classmethod
    def _find_valid_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[Any]],
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
                raise ValueError(
                    "quantization_encoding must be provided in pipeline_config"
                )

            kv_cache_size = cls._calculate_kv_cache_size(
                pipeline_model,
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

    @classmethod
    def _calculate_kv_cache_size(
        cls,
        pipeline_model: type[PipelineModel[Any]],
        pipeline_config: PipelineConfig,
        available_kv_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Calculate the KV cache size for the current configuration."""
        if issubclass(pipeline_model, KVCacheMixin):
            return pipeline_model.estimate_kv_cache_size(
                pipeline_config=pipeline_config,
                available_cache_memory=available_kv_cache_memory,
                devices=devices,
                huggingface_config=huggingface_config,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            )
        return 0

    @classmethod
    def _raise_oom_error(
        cls,
        pipeline_config: PipelineConfig,
        user_provided_max_length: bool,
        user_provided_max_batch_size: bool,
        pipeline_model: type[PipelineModel[Any]],
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
        ) = cls._find_valid_max_length(
            pipeline_config,
            pipeline_model,
            available_kv_cache_memory,
            user_provided_max_batch_size,
            huggingface_config,
            devices=devices,
        )

        pipeline_config.max_batch_size = original_max_batch_size

        found_valid_max_batch_size, inferred_max_batch_size = (
            cls._find_valid_batch_size(
                pipeline_config,
                pipeline_model,
                available_kv_cache_memory,
                original_max_length,
                user_provided_max_batch_size,
                huggingface_config,
                devices=devices,
            )
        )

        # Generate error message with suggestions
        error_msg = cls._generate_oom_error_message(
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

    @classmethod
    def _generate_oom_error_message(
        cls,
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
            cls._add_user_provided_max_length_suggestions(
                msg,
                user_provided_max_batch_size,
                found_valid_max_length,
                found_valid_max_batch_size,
                inferred_max_length,
                inferred_max_batch_size,
                inferred_max_length_compatible_batch_size,
            )
        else:
            cls._add_default_max_length_suggestions(
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

    @classmethod
    def _add_user_provided_max_length_suggestions(
        cls,
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

    @classmethod
    def _add_default_max_length_suggestions(
        cls,
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

    @classmethod
    def _infer_optimal_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[Any]],
        available_kv_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        return pipeline_model.infer_optimal_batch_size(
            pipeline_config,
            available_kv_cache_memory,
            huggingface_config=huggingface_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
