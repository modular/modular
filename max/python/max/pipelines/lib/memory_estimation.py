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
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, cast

from max.driver import Device, is_virtual_device_mode
from max.nn.kv_cache import (
    compute_max_seq_len_fitting_in_cache,
    estimated_memory_size,
)
from max.support.human_readable_formatter import to_human_readable_bytes

if TYPE_CHECKING:
    from max.pipelines.lib.registry import SupportedArchitecture

    from .config import PipelineConfig

from .config.model_config import MAXModelConfig
from .interfaces import (
    ArchConfig,
    ArchConfigWithKVCache,
)

logger = logging.getLogger("max.pipelines")

_DEFAULT_BATCH_SIZE = 512


@dataclass(frozen=True)
class _MemoryPlan:
    """Result of memory planning for a pipeline.

    Note: ``estimate_memory_footprint`` also mutates ``model_config`` directly
    (``kv_cache._available_cache_memory``, ``max_length``). Those mutations are
    the next thing to clean up as part of removing ``resolve()``.
    """

    max_batch_size: int
    footprint: int


# Vision encoder cache and paged token KV share the same pre-KV memory pool
# (see ``estimate_memory_footprint``). Without an explicit cap, reduction could
# assign almost the entire pool to vision, leaving insufficient memory for even
# one KV page. This bounds vision to a fraction so token KV always retains the rest.
_VISION_CACHE_MAX_FRACTION_OF_KV_BUDGET = 0.20


class MemoryEstimator:
    """Estimates available memory for pipeline model allocation."""

    @classmethod
    def free_memory(cls, devices: list[Device]) -> int:
        """Returns the total free memory available across all provided devices."""
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
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        signal_buffer_size: int = 0,
    ) -> int:
        """Calculates static memory usage: model weights plus activations plus signal buffers.

        Args:
            model_weights_size: Size of model weights.
            activation_memory_size: Size of activation memory.
            signal_buffer_size: Size of P2P signal buffers (fixed-size
                allocations used by collective comm kernels). Defaults to 0.

        Returns:
            Total static memory usage in bytes.
        """
        return model_weights_size + activation_memory_size + signal_buffer_size

    @classmethod
    def available_kv_cache_memory(
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        model_config: MAXModelConfig,
        devices: list[Device],
        signal_buffer_size: int = 0,
    ) -> int:
        """Estimates available KV cache memory after model weights, activations, and signal buffers.

        Args:
            model_weights_size: Size of model weights.
            activation_memory_size: Size of activation memory.
            model_config: The model configuration.
            devices: The list of devices on which the model will run.
            signal_buffer_size: Size of P2P signal buffers. Defaults to 0.

        Returns:
            Available KV cache memory in bytes.
        """
        return int(
            (
                cls.free_memory(devices)
                * model_config.kv_cache.device_memory_utilization
            )
            - cls.static_memory_size(
                model_weights_size,
                activation_memory_size,
                signal_buffer_size,
            )
        )

    @classmethod
    def max_supported_sequence_length(
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        model_config: MAXModelConfig,
        devices: list[Device],
        arch_config: ArchConfig,
        signal_buffer_size: int = 0,
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

        # Prefer the KV byte budget committed in ``estimate_memory_footprint`` (after
        # vision cache reservation and ``estimated_memory_size``). Using only
        # ``available_kv_cache_memory()`` (pre-vision) can overcount blocks and clamp
        # ``max_length`` above the physical paged KV capacity, causing runtime
        # InsufficientBlocksError when ``len(tokens)`` reaches ``total_blocks * page_size + 1``.
        allocated_kv = model_config.kv_cache._available_cache_memory
        if allocated_kv is not None:
            kvcache_mem = allocated_kv
        else:
            kvcache_mem = cls.available_kv_cache_memory(
                model_weights_size,
                activation_memory_size,
                model_config,
                devices,
                signal_buffer_size,
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
        signal_buffer_size: int = 0,
        arch: SupportedArchitecture | None = None,
        max_batch_size: int | None = None,
    ) -> _MemoryPlan:
        """Estimates memory footprint and validates ``max_length``/``max_batch_size`` fit.

        Returns:
            A :class:`_MemoryPlan` whose ``max_batch_size`` is the resolved
            value for this pipeline. Callers are responsible for storing and
            threading it; it is no longer written back to
            ``pipeline_config.runtime``.
        """
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
            max_batch_size = max_batch_size or 1
            if not model_config.max_length:
                model_config.max_length = arch_config.get_max_seq_len()
            # Set a large available cache memory value since we're not actually
            # allocating memory during cross-compilation. Use 1TB as a reasonable
            # large value that should work for any model.
            model_config.kv_cache._available_cache_memory = (
                1024 * 1024 * 1024 * 1024  # 1TB
            )
            return _MemoryPlan(max_batch_size=max_batch_size, footprint=0)

        try:
            free_memory = cls.free_memory(devices)
        except Exception:
            if is_draft_model:
                # Early return for draft model - we don't modify the original config
                return _MemoryPlan(
                    max_batch_size=max_batch_size or 1,
                    footprint=0,
                )
            resolved = max_batch_size or 1
            if not model_config.max_length:
                model_config.max_length = arch_config.get_max_seq_len()
            return _MemoryPlan(max_batch_size=resolved, footprint=0)

        # Total static memory requirement (weights + activations + signal buffers)
        static_memory_size = (
            model_weights_size + activation_memory_size + signal_buffer_size
        )

        if static_memory_size > free_memory:
            error_msg = f"Model size exceeds available memory ({to_human_readable_bytes(static_memory_size)} > {to_human_readable_bytes(free_memory)}). "
            if activation_memory_size > 0 or signal_buffer_size > 0:
                error_msg += (
                    f"Model weights: {to_human_readable_bytes(model_weights_size)}, "
                    f"Activation memory: {to_human_readable_bytes(activation_memory_size)}, "
                    f"Signal buffers: {to_human_readable_bytes(signal_buffer_size)}. "
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
                f"The model {to_human_readable_bytes(model_weights_size)}, activations "
                f"{to_human_readable_bytes(activation_memory_size)}, and signal buffers "
                f"{to_human_readable_bytes(signal_buffer_size)} don't leave room for KV cache. "
                f"Try running a smaller model, using a smaller precision, or using a device with more memory."
            )

        # KV cache is one buffer per device; budget can't exceed the
        # per-allocation cap (Metal's maxBufferLength).
        available_kv_cache_memory = min(
            available_kv_cache_memory,
            sum(d.max_single_alloc_size for d in devices),
        )

        vision_cache_bytes = cls._reserve_vision_cache_memory(
            pipeline_config,
            model_config,
            available_kv_cache_memory,
            devices,
            arch_config,
            arch=arch,
        )
        available_kv_cache_memory -= vision_cache_bytes
        total_size += vision_cache_bytes

        user_provided_max_length = model_config.max_length is not None
        user_provided_max_batch_size = max_batch_size is not None

        if is_draft_model:
            if not model_config.quantization_encoding:
                raise ValueError(
                    "quantization_encoding must be provided for draft model"
                )

            assert max_batch_size is not None, (
                "max_batch_size must be provided for draft model"
            )
            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
            )

            model_config.kv_cache._available_cache_memory = kv_cache_size

            return _MemoryPlan(
                max_batch_size=max_batch_size,
                footprint=int(total_size),
            )

        if not user_provided_max_length:
            model_config.max_length = arch_config.get_max_seq_len()

        if not model_config.quantization_encoding:
            raise ValueError(
                "quantization_encoding must be provided in pipeline_config"
            )

        if user_provided_max_batch_size:
            assert max_batch_size is not None
        else:
            max_batch_size = cls._infer_optimal_batch_size(arch_config, devices)
        if max_batch_size > pipeline_config.runtime.max_batch_input_tokens:
            logger.info(
                f"max_batch_size of {max_batch_size} cannot be larger than max_batch_input_tokens of {pipeline_config.runtime.max_batch_input_tokens}, overriding max_batch_size to {pipeline_config.runtime.max_batch_input_tokens}"
            )
            max_batch_size = pipeline_config.runtime.max_batch_input_tokens

        actual_kv_cache_size = cls._calculate_kv_cache_size(
            arch_config=arch_config,
            max_batch_size=max_batch_size,
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
                max_batch_size,
            ) = cls._find_valid_max_length(
                pipeline_config,
                arch_config,
                available_kv_cache_memory,
                user_provided_max_batch_size,
                max_batch_size,
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
                max_batch_size=max_batch_size,
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
                    max_batch_size,
                    total_size,
                    free_memory,
                    available_kv_cache_memory,
                    devices,
                )

            elif int(total_size) > int(vram_usage_limit_scale * free_memory):
                logger.warning(
                    "Estimated model and kv cache memory use nears available memory. You may experience errors."
                )

        return _MemoryPlan(
            max_batch_size=max_batch_size, footprint=int(total_size)
        )

    @classmethod
    def _find_valid_max_length(
        cls,
        pipeline_config: PipelineConfig,
        arch_config: ArchConfig,
        available_kv_cache_memory: int,
        user_provided_max_batch_size: bool,
        max_batch_size: int,
        devices: list[Device],
    ) -> tuple[bool, int, int]:
        """Binary search to find a valid max_length configuration.

        Returns:
            Tuple containing:
            - found_valid_max_length: Whether a valid max_length was found
            - inferred_max_length: The suggested max_length value
            - max_batch_size: The batch size used/inferred during the search
        """
        model_config = pipeline_config.model
        assert model_config.max_length is not None

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
                max_batch_size = cls._infer_optimal_batch_size(
                    arch_config, devices
                )

            # Use max_seq_len_override for binary search since we're varying model_config.max_length
            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=max_batch_size,
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
            max_batch_size,
        )

    @classmethod
    def _find_valid_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        available_kv_cache_memory: int,
        original_max_length: int,
        user_provided_max_batch_size: bool,
        max_batch_size: int,
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
        inferred_max_batch_size = max_batch_size
        lower = 1
        upper = max_batch_size

        while not found_valid_max_batch_size:
            inferred_max_batch_size = (lower + upper) // 2

            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=inferred_max_batch_size,
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
            return estimated_memory_size(
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
        max_batch_size: int,
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
            max_batch_size,
            devices,
        )

        found_valid_max_batch_size, inferred_max_batch_size = (
            cls._find_valid_batch_size(
                pipeline_config,
                available_kv_cache_memory,
                original_max_length,
                user_provided_max_batch_size,
                max_batch_size,
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
    def _reserve_vision_cache_memory(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
        available_memory: int,
        devices: list[Device],
        arch_config: ArchConfig,
        arch: SupportedArchitecture | None = None,
    ) -> int:
        """Estimate and reserve memory for the vision encoder cache.

        Delegates to ``arch.memory_planner.estimate_vision_cache_entry_bytes()``.
        Non-VLM architectures whose planner returns ``0`` reserve no vision
        cache memory.

        Vision cache is capped to at most a fraction of the shared KV+vision pool
        (see ``_VISION_CACHE_MAX_FRACTION_OF_KV_BUDGET``) so token KV cache retains
        the remainder. Entries may also be reduced when the pool is small.

        Returns:
            Bytes to reserve for the vision encoder cache (0 for non-VLM
            models or when ``max_vision_cache_entries`` is 0).
        """
        max_entries = pipeline_config.runtime.max_vision_cache_entries
        if max_entries <= 0:
            return 0

        hf_config = model_config.huggingface_config

        if arch is None:
            return 0

        if arch.memory_planner is None:
            return 0

        planner = arch.memory_planner(arch_config)
        per_entry_bytes = planner.estimate_vision_cache_entry_bytes(hf_config)
        if per_entry_bytes <= 0:
            return 0

        n_devices = len(devices)
        per_replica_bytes = per_entry_bytes * n_devices
        requested_bytes = max_entries * per_replica_bytes

        max_vision_bytes = int(
            available_memory * _VISION_CACHE_MAX_FRACTION_OF_KV_BUDGET
        )
        max_entries_budget = max_vision_bytes // per_replica_bytes
        effective_max_entries = min(max_entries, max_entries_budget)

        if effective_max_entries == 0:
            if max_entries > 0:
                logger.warning(
                    "Disabling vision encoder cache (requested %d entries, %s); "
                    "KV pool is too small to reserve vision entries within %.0f%% "
                    "of the pool (%s cap per entry %s).",
                    max_entries,
                    to_human_readable_bytes(requested_bytes),
                    _VISION_CACHE_MAX_FRACTION_OF_KV_BUDGET * 100,
                    to_human_readable_bytes(max_vision_bytes),
                    to_human_readable_bytes(per_replica_bytes),
                )
                pipeline_config.runtime.max_vision_cache_entries = 0
            return 0

        total_bytes = effective_max_entries * per_replica_bytes

        if effective_max_entries < max_entries:
            logger.warning(
                "Reduced vision encoder cache from %d (%s) to %d (%s) entries.",
                max_entries,
                to_human_readable_bytes(requested_bytes),
                effective_max_entries,
                to_human_readable_bytes(total_bytes),
            )
            pipeline_config.runtime.max_vision_cache_entries = (
                effective_max_entries
            )

        logger.info(
            "Vision encoder cache: %d entries, %s reserved.",
            pipeline_config.runtime.max_vision_cache_entries,
            to_human_readable_bytes(total_bytes),
        )

        return total_bytes

    @classmethod
    def _infer_optimal_batch_size(
        cls,
        arch_config: ArchConfig,
        devices: list[Device],
    ) -> int:
        """Infer the optimal batch size for the model.

        Args:
            arch_config: Architecture config that provides KV cache parameters.
            devices: The list of devices on which the model will run.
        """
        if not isinstance(arch_config, ArchConfigWithKVCache):
            return 1
        if len(devices) == 1 and devices[0].is_host:
            # batching on CPU is generally not useful, so we hard-code a batch size of 1.
            return 1
        return _DEFAULT_BATCH_SIZE
