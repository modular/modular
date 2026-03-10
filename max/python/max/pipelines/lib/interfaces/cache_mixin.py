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

"""Caching mixin for diffusion pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from max._core.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType

from .component_model import ComponentModel
from .diffusion_pipeline import max_compile


@dataclass
class CacheConfig:
    """Pipeline-level cache configuration, set once at init time."""

    step_cache: bool = False
    rdt: float | None = None
    """Relative difference threshold for cache reuse.  ``None`` means the
    pipeline should use its own model-appropriate default."""
    taylorseer: bool = False
    taylorseer_cache_interval: int = 5
    taylorseer_warmup_steps: int = 3
    taylorseer_max_order: int = 1


@dataclass
class DenoisingCacheState:
    """Per-request mutable cache state for a single denoising stream.

    One instance per stream (e.g. Flux1 true-CFG uses two: positive + negative).
    Created fresh per execute() call.
    """

    # FBCache state
    prev_residual: Tensor | None = None
    prev_output: Tensor | None = None

    # TaylorSeer state
    taylor_factor_0: Tensor | None = None
    taylor_factor_1: Tensor | None = None
    taylor_factor_2: Tensor | None = None
    taylor_last_compute_step: int | None = None


class CacheMixin:
    """Mixin providing caching support for diffusion pipelines.

    Subclasses call ``init_cache(...)`` during their
    ``init_remaining_components()`` to configure caching once at pipeline
    construction time.
    """

    cache_config: CacheConfig
    _gpu_scalar_cache: dict[tuple[Any, float], Tensor]

    # Pre-allocated tensors (created once at init, reused across requests)
    _cache_step_cache_flag: Tensor | None
    _cache_rdt_tensor: Tensor | None
    _cache_taylor_max_order_tensor: Tensor | None

    _cache_dtype: DType
    _cache_device: Device

    def init_cache(
        self,
        cache_config: CacheConfig,
        transformer: ComponentModel,
        dtype: DType,
        device: Device,
        default_rdt: float = 0.05,
    ) -> None:
        """Initialize caching subsystem. Call once during init_remaining_components().

        This method:
        1. Stores the cache config.
        2. Selects and compiles the correct transformer graph variant.
        3. Pre-allocates constant tensors.
        4. Stores dtype/device for per-request DenoisingCacheState creation.
        5. Builds TaylorSeer compiled graphs if enabled.

        Args:
            default_rdt: Model-specific default for the relative difference
                threshold.  Used when ``cache_config.rdt`` is ``None``.
        """
        self.cache_config = cache_config
        self._gpu_scalar_cache = {}

        rdt = cache_config.rdt if cache_config.rdt is not None else default_rdt

        # Graph selection (init-time, not per-request)
        if cache_config.step_cache:
            transformer.use_step_cache_model()
        else:
            transformer.use_standard_model()

        # Pre-allocate constant tensors
        self._cache_step_cache_flag = None
        self._cache_rdt_tensor = None
        if cache_config.step_cache:
            self._cache_step_cache_flag = Tensor.full(
                [1],
                True,
                dtype=DType.bool,
                device=device,
            )
            self._cache_rdt_tensor = Tensor.full(
                [1],
                rdt,
                dtype=DType.float32,
                device=device,
            )

        self._cache_taylor_max_order_tensor = None
        if cache_config.taylorseer:
            self._cache_taylor_max_order_tensor = Tensor.full(
                [1],
                cache_config.taylorseer_max_order,
                dtype=DType.int32,
                device=device,
            )

        self._cache_dtype = dtype
        self._cache_device = device

        # Build TaylorSeer graphs if enabled
        if cache_config.taylorseer:
            self.build_taylorseer(dtype, device)

    def create_cache_state(
        self,
        batch_size: int,
        seq_len: int,
        transformer_config: Any,
    ) -> DenoisingCacheState:
        """Create per-request cache state with fresh tensors.

        Args:
            batch_size: Batch dimension (from prompt_embeds).
            seq_len: Sequence length (from latents).
            transformer_config: Transformer config carrying dimension info.
                Must have ``num_attention_heads``, ``attention_head_dim``,
                ``patch_size``, ``out_channels``, and ``in_channels`` attributes.
        """
        for attr in (
            "num_attention_heads",
            "attention_head_dim",
            "patch_size",
            "out_channels",
            "in_channels",
        ):
            assert hasattr(transformer_config, attr), (
                f"transformer_config missing required attribute '{attr}'"
            )

        residual_dim = (
            transformer_config.num_attention_heads
            * transformer_config.attention_head_dim
        )
        output_dim = (
            transformer_config.patch_size
            * transformer_config.patch_size
            * (transformer_config.out_channels or transformer_config.in_channels)
        )

        state = DenoisingCacheState()

        if self.cache_config.step_cache:
            state.prev_residual = Tensor.zeros(
                (batch_size, seq_len, residual_dim),
                dtype=self._cache_dtype,
                device=self._cache_device,
            )
            state.prev_output = Tensor.zeros(
                (batch_size, seq_len, output_dim),
                dtype=self._cache_dtype,
                device=self._cache_device,
            )

        if self.cache_config.taylorseer:
            for attr in ("taylor_factor_0", "taylor_factor_1", "taylor_factor_2"):
                setattr(
                    state,
                    attr,
                    Tensor.zeros(
                        (batch_size, seq_len, output_dim),
                        dtype=self._cache_dtype,
                        device=self._cache_device,
                    ),
                )

        return state

    def _make_gpu_scalar(self, value: float, device: Device) -> Tensor:
        """Return a cached [1]-element float32 GPU tensor."""
        key = (device, value)
        t = self._gpu_scalar_cache.get(key)
        if t is None:
            t = Tensor.full([1], value, dtype=DType.float32, device=device)
            self._gpu_scalar_cache[key] = t
        return t

    def build_taylorseer(self, dtype: DType, device: Device) -> None:
        """Build compiled graphs for TaylorSeer predict and update."""
        tensor_type = TensorType(
            dtype, shape=["batch", "seq", "channels"], device=device
        )
        scalar_type = TensorType(DType.float32, shape=[1], device=device)
        order_type = TensorType(DType.int32, shape=[1], device=device)

        self.__dict__["taylor_predict"] = max_compile(
            self.taylor_predict,
            input_types=[
                tensor_type,  # factor_0
                tensor_type,  # factor_1
                tensor_type,  # factor_2
                scalar_type,  # step_offset
                order_type,  # max_order
            ],
        )
        self.__dict__["taylor_update"] = max_compile(
            self.taylor_update,
            input_types=[
                tensor_type,  # new_output
                tensor_type,  # old_factor_0
                tensor_type,  # old_factor_1
                scalar_type,  # delta_step
                order_type,  # max_order
            ],
        )

    @staticmethod
    def taylor_predict(
        factor_0: Tensor,
        factor_1: Tensor,
        factor_2: Tensor,
        step_offset: Tensor,
        max_order: Tensor,
    ) -> Tensor:
        """Taylor series prediction: f(t+dt) ~ f(t) + f'(t)*dt + f''(t)*dt^2/2."""
        offset = F.cast(step_offset, factor_0.dtype)
        result = factor_0 + factor_1 * offset
        offset_sq_half = offset * offset * F.constant(
            0.5, factor_0.dtype, device=factor_0.device
        )
        order2_term = factor_2 * offset_sq_half
        use_order2 = max_order >= F.constant(
            2, DType.int32, device=max_order.device
        )
        use_order2_cast = F.cast(
            F.broadcast_to(use_order2, order2_term.shape), order2_term.dtype
        )
        result = result + order2_term * use_order2_cast
        return result

    @staticmethod
    def taylor_update(
        new_output: Tensor,
        old_factor_0: Tensor,
        old_factor_1: Tensor,
        delta_step: Tensor,
        max_order: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Taylor factors via divided differences."""
        delta = F.cast(delta_step, new_output.dtype)
        eps = F.constant(1e-9, new_output.dtype, device=new_output.device)
        safe_delta = delta + eps

        new_factor_0 = new_output
        new_factor_1 = (new_output - old_factor_0) / safe_delta
        new_factor_2 = (new_factor_1 - old_factor_1) / safe_delta
        use_order2 = max_order >= F.constant(
            2, DType.int32, device=max_order.device
        )
        use_order2_cast = F.cast(
            F.broadcast_to(use_order2, new_factor_2.shape), new_factor_2.dtype
        )
        new_factor_2 = new_factor_2 * use_order2_cast

        return new_factor_0, new_factor_1, new_factor_2

    @staticmethod
    def taylorseer_skip_transformer(
        step: int, warmup_steps: int, cache_interval: int
    ) -> bool:
        """Return True when a full transformer pass is needed at *step*."""
        if step < warmup_steps:
            return False
        return (step - warmup_steps - 1) % cache_interval != 0


def can_use_step_cache(
    intermediate_residual: Tensor,
    prev_intermediate_residual: Tensor | None,
    rdt: Tensor,
) -> Tensor:
    """Return whether previous residual cache is reusable (RDT check)."""
    dev = intermediate_residual.device
    if (
        prev_intermediate_residual is None
        or intermediate_residual.shape != prev_intermediate_residual.shape
    ):
        return F.constant(False, DType.bool, device=dev)

    mean_diff_rows = F.mean(
        F.abs(intermediate_residual - prev_intermediate_residual), axis=-1
    )
    mean_prev_rows = F.mean(F.abs(prev_intermediate_residual), axis=-1)
    mean_diff = F.mean(mean_diff_rows, axis=None)
    mean_prev = F.mean(mean_prev_rows, axis=None)
    eps = 1e-9
    relative_diff = mean_diff / (mean_prev + eps)
    pred = relative_diff < F.cast(rdt, relative_diff.dtype)
    return F.squeeze(pred, 0)
