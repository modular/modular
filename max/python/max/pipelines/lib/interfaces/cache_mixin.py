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

"""Caching types and utilities for diffusion pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from max.config import ConfigFileModel
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType, TensorValue
from pydantic import ConfigDict, Field, model_validator


class DenoisingCacheConfig(ConfigFileModel):
    """Pipeline-level cache configuration for diffusion model denoising.

    Controls First-Block Cache (step cache) and TaylorSeer optimizations
    that skip redundant transformer passes during the denoising loop.
    """

    model_config = ConfigDict(frozen=False)

    first_block_caching: bool = Field(
        default=False,
        description=(
            "Enable First-Block Cache (FBCache) for step-cache denoising. "
            "When enabled, the transformer skips remaining blocks if the "
            "first-block residual is similar to the previous step."
        ),
    )

    taylorseer: bool = Field(
        default=False,
        description=(
            "Enable TaylorSeer cache optimization. Uses Taylor series "
            "prediction to skip full transformer passes on certain "
            "denoising steps."
        ),
    )

    taylorseer_cache_interval: int | None = Field(
        default=None,
        description=(
            "Steps between full TaylorSeer computations. "
            "None uses the model-specific default (typically 5)."
        ),
    )

    taylorseer_warmup_steps: int | None = Field(
        default=None,
        description=(
            "Number of warmup steps before TaylorSeer prediction begins. "
            "None uses the model-specific default (typically 4)."
        ),
    )

    taylorseer_max_order: int | None = Field(
        default=None,
        description=(
            "Taylor expansion order (1 or 2). Higher order uses second "
            "derivatives for more accurate prediction. "
            "None uses the model-specific default (typically 1)."
        ),
    )

    teacache: bool = Field(
        default=False,
        description=(
            "Enable TeaCache cache optimization. Uses the timestep-aware "
            "modulated input change to decide when the FLUX.2 transformer "
            "backbone can be skipped."
        ),
    )

    teacache_rel_l1_thresh: float | None = Field(
        default=None,
        description=(
            "Relative-L1 threshold used by TeaCache. "
            "None uses the model-specific default."
        ),
    )

    teacache_coefficients: list[float] | None = Field(
        default=None,
        description=(
            "Polynomial coefficients used to rescale TeaCache's relative-L1 "
            "metric. None uses the model-specific default coefficients."
        ),
    )

    @model_validator(mode="after")
    def _validate_cache_mode(self) -> DenoisingCacheConfig:
        if (
            self.taylorseer_cache_interval is not None
            and self.taylorseer_cache_interval < 1
        ):
            raise ValueError("taylorseer_cache_interval must be >= 1.")
        if (
            self.taylorseer_warmup_steps is not None
            and self.taylorseer_warmup_steps < 1
        ):
            raise ValueError("taylorseer_warmup_steps must be >= 1.")
        if self.taylorseer_max_order is not None and (
            self.taylorseer_max_order not in (1, 2)
        ):
            raise ValueError("taylorseer_max_order must be 1 or 2.")
        if self.teacache and self.first_block_caching:
            raise ValueError(
                "TeaCache cannot be enabled together with first_block_caching."
            )
        if self.teacache and self.taylorseer:
            raise ValueError(
                "TeaCache cannot be enabled together with taylorseer."
            )
        if (
            self.teacache_rel_l1_thresh is not None
            and self.teacache_rel_l1_thresh < 0.0
        ):
            raise ValueError("teacache_rel_l1_thresh must be non-negative.")
        if self.teacache_coefficients is not None and (
            len(self.teacache_coefficients) < 1
        ):
            raise ValueError(
                "teacache_coefficients must contain at least 1 coefficient."
            )
        return self


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

    # TeaCache state
    teacache_prev_modulated_input: Tensor | None = None
    teacache_cached_residual: Tensor | None = None
    teacache_accumulated_rel_l1: Tensor | None = None


def fbcache_conditional_execution(
    first_block_residual: Tensor,
    prev_residual: Tensor,
    prev_output: Tensor,
    residual_threshold: Tensor,
    run_remaining_blocks: Callable[..., Tensor],
    run_remaining_kwargs: dict[str, Any],
    run_postamble: Callable[[Tensor, Tensor], Tensor],
    temb: Tensor,
    output_types: list[TensorType],
) -> tuple[Tensor, Tensor]:
    """Handle FBCache F.cond branching pattern shared across DiT models.

    The caller provides atomic DiT methods:
    - *run_remaining_blocks*: runs blocks 1..N + single-stream blocks,
      returns pre-tail hidden states.
    - *run_postamble*: applies final norm + projection.

    The ``residual_threshold`` is a scalar Tensor (float32, shape=[]) passed
    as a graph input so it can be changed at runtime without recompilation.

    Returns:
        (first_block_residual, output) tensors.
    """
    use_fbcache = can_use_fbcache(
        first_block_residual, prev_residual, residual_threshold
    )

    def then_fn(
        _prev_output: Tensor = prev_output,
        _first_block_residual: Tensor = first_block_residual,
    ) -> tuple[TensorValue, TensorValue]:
        return (
            TensorValue(_first_block_residual),
            TensorValue(_prev_output),
        )

    def else_fn(
        _fbr: Tensor = first_block_residual,
    ) -> tuple[TensorValue, TensorValue]:
        hidden_states = run_remaining_blocks(**run_remaining_kwargs)
        out = run_postamble(hidden_states, temb)
        return (TensorValue(_fbr), TensorValue(out))

    result = F.cond(
        use_fbcache,
        output_types,
        then_fn,
        else_fn,
    )
    return (result[0], result[1])


def can_use_fbcache(
    intermediate_residual: Tensor,
    prev_intermediate_residual: Tensor | None,
    residual_threshold: Tensor,
) -> Tensor:
    """Return whether previous residual cache is reusable (RDT check).

    Args:
        intermediate_residual: First-block residual for the current step.
        prev_intermediate_residual: First-block residual from the previous step.
        residual_threshold: Scalar Tensor (float32, shape=[]) with the
            relative difference threshold.  Passed as a graph input so it
            can be changed at runtime without recompilation.
    """
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
    rdt = residual_threshold.cast(relative_diff.dtype)
    pred = relative_diff < rdt
    return F.squeeze(pred, 0)


def teacache_rescaled_delta(
    modulated_input: Tensor,
    prev_modulated_input: Tensor,
    coefficients: tuple[float, ...],
) -> Tensor:
    """Compute the polynomial-rescaled relative-L1 delta for TeaCache.

    Uses Horner's method to evaluate the polynomial defined by *coefficients*
    (descending degree order, matching ``np.poly1d`` convention) on the
    relative-L1 distance between consecutive modulated inputs.

    Args:
        modulated_input: Current timestep-modulated hidden states.
        prev_modulated_input: Previous step's modulated hidden states.
        coefficients: Polynomial coefficients in descending degree order.

    Returns:
        Scalar float32 tensor (shape ``[1]``) with the rescaled delta.
    """
    mean_diff_rows = F.mean(
        F.abs(modulated_input - prev_modulated_input), axis=-1
    )
    mean_prev_rows = F.mean(F.abs(prev_modulated_input), axis=-1)
    mean_diff = F.mean(mean_diff_rows, axis=None)
    mean_prev = F.mean(mean_prev_rows, axis=None)
    eps = F.constant(1e-9, mean_diff.dtype, device=mean_diff.device)
    relative_diff = mean_diff / (mean_prev + eps)
    relative_diff_f32 = F.cast(relative_diff, DType.float32)

    # Horner's method: ((((c0 * x + c1) * x + c2) * x + c3) * x + c4)
    x = F.constant(coefficients[0], DType.float32, device=relative_diff.device)
    for coeff in coefficients[1:]:
        coeff_tensor = F.constant(coeff, DType.float32, device=x.device)
        x = x * relative_diff_f32 + coeff_tensor
    return F.reshape(x, [1])


def teacache_conditional_execution(
    modulated_input: Tensor,
    next_accumulated: Tensor,
    accumulated_rel_l1: Tensor,
    force_compute: Tensor,
    rel_l1_thresh: float,
    projected_hidden_states: Tensor,
    prev_residual: Tensor,
    temb: Tensor,
    run_first_block: Callable[..., tuple[Tensor, Tensor]],
    first_block_kwargs: dict[str, Any],
    run_remaining_blocks: Callable[..., Tensor],
    remaining_blocks_kwargs: dict[str, Any],
    run_postamble: Callable[[Tensor, Tensor], Tensor],
    output_types: list[TensorType],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Handle TeaCache F.cond branching pattern shared across DiT models.

    Parallel to ``fbcache_conditional_execution``: the shared utility
    constructs the then/else closures internally.

    Args:
        modulated_input: Current step's timestep-modulated proxy signal.
        next_accumulated: Accumulated rescaled delta including this step.
        accumulated_rel_l1: Accumulated rescaled delta before this step
            (used to produce the zero-reset value on compute steps).
        force_compute: Scalar bool tensor.  When True (first/last step),
            the transformer always runs regardless of the accumulated delta.
        rel_l1_thresh: Threshold for skipping; skip when accumulated < thresh.
        projected_hidden_states: Output of ``x_embedder`` (pre-block input).
        prev_residual: Cached residual from the last compute step.
        temb: Timestep embedding for the current step.
        run_first_block: Runs dual-stream block 0, returns
            ``(encoder_hidden, image_hidden)``.
        first_block_kwargs: Keyword arguments forwarded to *run_first_block*.
        run_remaining_blocks: Runs dual-stream blocks 1..N + single-stream
            blocks, returns pre-postamble image hidden states.
        remaining_blocks_kwargs: Keyword arguments forwarded to
            *run_remaining_blocks* (excluding ``hidden_states`` and
            ``encoder_hidden_states``, which come from *run_first_block*).
        run_postamble: Applies final norm + projection,
            ``(hidden_states, temb) -> output``.
        output_types: Tensor types for ``F.cond`` output specification.

    Returns:
        ``(modulated_input, residual_or_prev, accumulated_rel_l1, output)``
    """
    thresh = F.constant(
        rel_l1_thresh, DType.float32, device=next_accumulated.device
    )
    should_skip = F.squeeze(~force_compute & (next_accumulated < thresh), 0)

    def then_fn() -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        output = run_postamble(projected_hidden_states + prev_residual, temb)
        return (
            TensorValue(modulated_input),
            TensorValue(prev_residual),
            TensorValue(next_accumulated),
            TensorValue(output),
        )

    def else_fn() -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        first_encoder, first_hidden = run_first_block(**first_block_kwargs)
        hidden_states = run_remaining_blocks(
            hidden_states=first_hidden,
            encoder_hidden_states=first_encoder,
            **remaining_blocks_kwargs,
        )
        residual = hidden_states - projected_hidden_states
        output = run_postamble(hidden_states, temb)
        zero_accumulated = accumulated_rel_l1 - accumulated_rel_l1
        return (
            TensorValue(modulated_input),
            TensorValue(residual),
            TensorValue(zero_accumulated),
            TensorValue(output),
        )

    result = F.cond(should_skip, output_types, then_fn, else_fn)
    return (result[0], result[1], result[2], result[3])
