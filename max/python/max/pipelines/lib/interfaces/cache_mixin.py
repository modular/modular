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
from pydantic import ConfigDict, Field


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


def fbcache_conditional_execution(
    first_block_residual: Tensor,
    prev_residual: Tensor,
    prev_output: Tensor,
    residual_threshold: Tensor,
    run_remaining_blocks: Callable[..., Tensor],
    run_remaining_kwargs: dict[str, Any],
    output_types: list[TensorType],
) -> tuple[Tensor, Tensor]:
    """Handle FBCache F.cond branching pattern shared across DiT models.

    This is only called from the step-cache graph path (where step-cache is
    always enabled), so there is no outer ``F.cond`` on a cache-enabled flag.
    The single ``F.cond`` checks the RDT (relative difference threshold) to
    decide whether to reuse the cached output or run the remaining blocks.

    The caller provides:
    - first_block_residual: computed as ``new_hidden_states - hidden_states``
      after running the first transformer block.
    - residual_threshold: scalar Tensor (float32, shape=[]) with the
      relative difference threshold.  Passed as a graph input so it can be
      changed at runtime without recompilation.
    - run_remaining_blocks: model-specific callable that runs remaining
      blocks + norm + proj.  Called with ``**run_remaining_kwargs`` and must
      return a single output ``Tensor``.
    - run_remaining_kwargs: keyword arguments forwarded to
      *run_remaining_blocks*.
    - output_types: ``[residual_type, output_type]`` passed directly to
      ``F.cond``.

    Returns:
        (first_block_residual, output) tensors.
    """
    use_step_cache = can_use_step_cache(
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
        out = run_remaining_blocks(**run_remaining_kwargs)
        return (TensorValue(_fbr), TensorValue(out))

    result = F.cond(
        use_step_cache,
        output_types,
        then_fn,
        else_fn,
    )
    return (result[0], result[1])


def can_use_step_cache(
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
