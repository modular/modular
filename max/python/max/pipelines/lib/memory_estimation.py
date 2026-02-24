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

import logging
from io import StringIO
from typing import TYPE_CHECKING, cast

from max.driver import Device, is_virtual_device_mode, load_devices
from max.kv_cache import estimate_kv_cache_size, infer_optimal_batch_size
from max.nn.kv_cache import compute_max_seq_len_fitting_in_cache
from max.support.human_readable_formatter import to_human_readable_bytes

if TYPE_CHECKING:
    from .config import PipelineConfig

from .interfaces import ArchConfig, ArchConfigWithKVCache
from .model_config import MAXModelConfig

logger = logging.getLogger("max.pipelines")

_MAX_DEFAULT_BATCH_SIZE = 4096
_MIN_DEFAULT_BATCH_SIZE = 1


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
    def static_memory_size(
        cls, model_weights_size: int, activation_memory_size: int
    ) -> int:
        """Calculates static memory usage: model weights plus activations.

        Args:
            model_weights_size: Size of model weights.
            activation_memory_size: Size of activation memory.

        Returns:
            Total static memory usage in bytes.
        """
        return model_weights_size + activation_memory_size

    @classmethod
    def available_kv_cache_memory(
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        model_config: MAXModelConfig,
        devices: list[Device],
    ) -> int:
        """Estimates available KV cache memory after model weights and activations.

        Args:
            model_weights_size: Size of model weights.
            activation_memory_size: Size of activation memory.
            model_config: The model configuration.
            devices: The list of devices on which the model will run.

        Returns:
            Available KV cache memory in bytes.
        """
        return int(
            (
                cls.free_memory(devices)
                * model_config.kv_cache.device_memory_utilization
            )
            - cls.static_memory_size(model_weights_size, activation_memory_size)
        )

    @classmethod
    def max_supported_sequence_length(
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        model_config: MAXModelConfig,
        devices: list[Device],
        arch_config: ArchConfig,
    ) -> int | None:
        """Computes the hard upper bound on tokens for a single request.

        Mirrors the paged KV cache constraint: per replica, a request cannot
        exceed total pages per device times page size.
        """
        # In virtual device mode (cross-compilation), skip memory-based constraints
        # since we're only compiling and not actually running the model.
        if is_virtual_device_mode():
            logger.info(
                "Skipping memory-based sequence length constraints in "
                "virtual device mode (cross-compilation)"
            )
            return None

        # Retrieve needed parameters.
        if not model_config.quantization_encoding:
            raise ValueError(
                "quantization_encoding must be provided in model_config"
            )

        if not isinstance(arch_config, ArchConfigWithKVCache):
            return None

        arch_config = cast(ArchConfigWithKVCache, arch_config)
        params = arch_config.get_kv_params()

        kvcache_mem = cls.available_kv_cache_memory(
            model_weights_size, activation_memory_size, model_config, devices
        )
        return compute_max_seq_len_fitting_in_cache(
            params=params,
            available_cache_memory=kvcache_mem,
        )

    @classmethod
    def estimate_memory_footprint(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
        arch_config: ArchConfig,
        devices: list[Device],
        model_weights_size: int,
        activation_memory_size: int,
    ) -> None:
        """Estimates memory footprint and validates max_length/max_batch_size fit."""
        is_draft_model = (
            pipeline_config.draft_model is not None
            and model_config is pipeline_config.draft_model
        )

        # In virtual device mode (cross-compilation), skip memory estimation
        # since we're only compiling and not actually running the model.
        # Use model defaults for max_batch_size and max_length.
        if is_virtual_device_mode():
            logger.info(
                "Skipping memory estimation in virtual device mode "
                "(cross-compilation)"
            )
            if not pipeline_config.max_batch_size:
                pipeline_config.max_batch_size = 1
            if not model_config.max_length:
                model_config.max_length = arch_config.get_max_seq_len()
            # Set a large available cache memory value since we're not actually
            # allocating memory during cross-compilation. Use 1TB as a reasonable
            # large value that should work for any model.
            model_config.kv_cache._available_cache_memory = (
                1024 * 1024 * 1024 * 1024  # 1TB
            )
            return

        try:
            free_memory = cls.free_memory(devices)
        except Exception as e:
            if is_draft_model:
                # Early return for draft model - we don't modify the original config
                return
            if not pipeline_config.max_batch_size:
                pipeline_config.max_batch_size = 1
            if not model_config.max_length:
                model_config.max_length = arch_config.get_max_seq_len()
            return

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
            free_memory * model_config.kv_cache.device_memory_utilization
            - static_memory_size
        )

        if available_kv_cache_memory <= 0:
            raise RuntimeError(
                f"The model {to_human_readable_bytes(model_weights_size)} and activations "
                f"{to_human_readable_bytes(activation_memory_size)} don't leave room for KV cache. "
                f"Try running a smaller model, using a smaller precision, or using a device with more memory."
            )

        user_provided_max_length = model_config.max_length is not None
        user_provided_max_batch_size = (
            pipeline_config.max_batch_size is not None
        )

        if is_draft_model:
            if not model_config.quantization_encoding:
                raise ValueError(
                    "quantization_encoding must be provided for draft model"
                )

            assert pipeline_config.max_batch_size is not None, (
                "max_batch_size must be provided for draft model"
            )
            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=pipeline_config.max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
            )

            model_config.kv_cache._available_cache_memory = kv_cache_size

            return  # Don't modify pipeline config values

        if not user_provided_max_length:
            model_config.max_length = arch_config.get_max_seq_len()

        if not model_config.quantization_encoding:
            raise ValueError(
                "quantization_encoding must be provided in pipeline_config"
            )

        if not user_provided_max_batch_size:
            pipeline_config.max_batch_size = cls._infer_optimal_batch_size(
                arch_config, available_kv_cache_memory, devices
            )

        assert pipeline_config.max_batch_size is not None
        if (
            pipeline_config.max_batch_size
            > pipeline_config.max_batch_input_tokens
        ):
            logger.info(
                f"max_batch_size of {pipeline_config.max_batch_size} cannot be larger than max_batch_input_tokens of {pipeline_config.max_batch_input_tokens}, overriding max_batch_size to {pipeline_config.max_batch_input_tokens}"
            )
            pipeline_config.max_batch_size = (
                pipeline_config.max_batch_input_tokens
            )

        actual_kv_cache_size = cls._calculate_kv_cache_size(
            arch_config=arch_config,
            max_batch_size=pipeline_config.max_batch_size,
            available_kv_cache_memory=available_kv_cache_memory,
        )

        model_config.kv_cache._available_cache_memory = actual_kv_cache_size

        total_size += actual_kv_cache_size
        # If the model is too large to fit in memory, and the user did not
        # specify a max_length, try to infer a value that would fit.
        if int(total_size) > free_memory and not user_provided_max_length:
            original_max_length = model_config.max_length
            (
                found_valid_max_length,
                inferred_max_length,
                _,
            ) = cls._find_valid_max_length(
                pipeline_config,
                arch_config,
                available_kv_cache_memory,
                user_provided_max_batch_size,
                devices=devices,
            )

            if found_valid_max_length:
                logger.warning(
                    f"Truncated model's default max_length from {original_max_length} to {inferred_max_length} to fit in memory."
                )
                model_config.max_length = inferred_max_length
            else:
                model_config.max_length = 1

            actual_kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=pipeline_config.max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
            )
            total_size = model_weights_size + actual_kv_cache_size

        vram_usage_limit_scale = 0.95

        if isinstance(free_memory, int | float):
            if int(total_size) > int(free_memory):
                cls._raise_oom_error(
                    pipeline_config,
                    arch_config,
                    user_provided_max_length,
                    user_provided_max_batch_size,
                    total_size,
                    free_memory,
                    available_kv_cache_memory,
                    devices,
                )

            elif int(total_size) > int(vram_usage_limit_scale * free_memory):
                logger.warning(
                    "Estimated model and kv cache memory use nears available memory. You may experience errors."
                )

    @classmethod
    def _find_valid_max_length(
        cls,
        pipeline_config: PipelineConfig,
        arch_config: ArchConfig,
        available_kv_cache_memory: int,
        user_provided_max_batch_size: bool,
        devices: list[Device],
    ) -> tuple[bool, int, int]:
        """Binary search to find a valid max_length configuration.

        Returns:
            Tuple containing:
            - found_valid_max_length: Whether a valid max_length was found
            - inferred_max_length: The suggested max_length value
            - inferred_max_length_compatible_batch_size: Compatible batch size for the max_length
        """
        model_config = pipeline_config.model
        assert model_config.max_length is not None
        assert pipeline_config.max_batch_size is not None

        found_valid_max_length = False
        lower = 1
        upper = model_config.max_length
        inferred_max_length = upper

        if not model_config.quantization_encoding:
            raise ValueError(
                "quantization_encoding must be provided in pipeline_config"
            )

        while not found_valid_max_length:
            inferred_max_length = (lower + upper) // 2
            model_config.max_length = inferred_max_length

            if not user_provided_max_batch_size:
                pipeline_config.max_batch_size = cls._infer_optimal_batch_size(
                    arch_config, available_kv_cache_memory, devices
                )

            # Use max_seq_len_override for binary search since we're varying model_config.max_length
            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=pipeline_config.max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
                max_seq_len_override=inferred_max_length,
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
        available_kv_cache_memory: int,
        original_max_length: int,
        user_provided_max_batch_size: bool,
        arch_config: ArchConfig,
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
        pipeline_config.model.max_length = original_max_length
        inferred_max_batch_size = cast(int, pipeline_config.max_batch_size)
        lower = 1
        upper = cast(int, pipeline_config.max_batch_size)

        while not found_valid_max_batch_size:
            inferred_max_batch_size = (lower + upper) // 2
            pipeline_config.max_batch_size = inferred_max_batch_size

            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=pipeline_config.max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
                max_seq_len_override=original_max_length,
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
        arch_config: ArchConfig,
        max_batch_size: int,
        available_kv_cache_memory: int,
        max_seq_len_override: int | None = None,
    ) -> int:
        """Calculate the KV cache size for the current configuration.

        Args:
            arch_config: Architecture config that potentially provides KV cache
                parameters.
            max_batch_size: The maximum batch size.
            available_kv_cache_memory: Available memory for KV cache in bytes.
            max_seq_len_override: Optional override for max sequence length.
                If provided, this value is used instead of querying arch_config.
                Useful during binary search over max_length.
        """
        if isinstance(arch_config, ArchConfigWithKVCache):
            params = arch_config.get_kv_params()
            max_seq_len = (
                max_seq_len_override
                if max_seq_len_override is not None
                else arch_config.get_max_seq_len()
            )
            return estimate_kv_cache_size(
                params=params,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                available_cache_memory=available_kv_cache_memory,
            )
        else:
            return 0

    @classmethod
    def _raise_oom_error(
        cls,
        pipeline_config: PipelineConfig,
        arch_config: ArchConfig,
        user_provided_max_length: bool,
        user_provided_max_batch_size: bool,
        total_size: int,
        original_free_memory: int,
        available_kv_cache_memory: int,
        devices: list[Device],
    ) -> None:
        """Suggests a viable configuration when the current one does not fit in memory.

        If the current configuration won't fit in device memory, provides a
        friendly error message. The approach is to:

        1. Binary search max_length until we find a setting that works
        2. If user provided max_batch_size, binary search that too
        3. Generate appropriate suggestions based on this truth table:

        .. code-block:: text

                                                                max_length
                                             +----------------------+--------------------------+
                                             | set by user          | set to default           |
                            +----------------+======================+==========================+
                            | set by user    ║ Recommend both       | Recommend max_batch_size |
            max_batch_size  +----------------+----------------------+--------------------------+
                            | set to default ║ Recommend max_length | Recommend both           |
                            +----------------+----------------------+--------------------------+
        """
        original_max_length = cast(int, pipeline_config.model.max_length)
        original_max_batch_size = cast(int, pipeline_config.max_batch_size)

        # Find valid configurations through binary search
        (
            found_valid_max_length,
            inferred_max_length,
            inferred_max_length_compatible_batch_size,
        ) = cls._find_valid_max_length(
            pipeline_config,
            arch_config,
            available_kv_cache_memory,
            user_provided_max_batch_size,
            devices,
        )

        pipeline_config.max_batch_size = original_max_batch_size

        found_valid_max_batch_size, inferred_max_batch_size = (
            cls._find_valid_batch_size(
                pipeline_config,
                available_kv_cache_memory,
                original_max_length,
                user_provided_max_batch_size,
                arch_config=arch_config,
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
        arch_config: ArchConfig,
        available_kv_cache_memory: int,
        devices: list[Device],
    ) -> int:
        """Infer the optimal batch size for the model.

        Args:
            arch_config: Architecture config that provides KV cache parameters.
            devices: The list of devices on which the model will run.
            available_kv_cache_memory: Available memory for KV cache in bytes.
        """
        if not isinstance(arch_config, ArchConfigWithKVCache):
            return _MIN_DEFAULT_BATCH_SIZE
        if len(devices) == 1 and devices[0].is_host:
            # batching on CPU is generally not useful, so we hard-code a batch size of 1.
            return 1

        kv_params = arch_config.get_kv_params()
        max_seq_len = arch_config.get_max_seq_len()

        device_objs = (
            devices
            if isinstance(devices[0], Device)
            else load_devices([d for d in devices])
        )

        inferred_batch_size = infer_optimal_batch_size(
            params=kv_params,
            max_seq_len=max_seq_len,
            available_cache_memory=available_kv_cache_memory,
            devices=device_objs,
        )
        # Clamp the batch size
        return max(
            _MIN_DEFAULT_BATCH_SIZE,
            min(inferred_batch_size, _MAX_DEFAULT_BATCH_SIZE),
        )
