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
        return self


@dataclass
class DenoisingCacheState:
    """Per-request mutable cache state for a single denoising stream.

    One instance per stream. Created fresh per execute() call.
    """

    # FBCache state
    prev_residual: Tensor | None = None
    prev_output: Tensor | None = None

    # TaylorSeer state
    taylor_factor_0: Tensor | None = None
    taylor_factor_1: Tensor | None = None
    taylor_factor_2: Tensor | None = None
    taylor_last_compute_step: int | None = None


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
