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

"""Standalone TaylorSeer caching module for diffusion pipelines.

TaylorSeer uses Taylor series approximation to skip full transformer
passes during denoising.  On steps where the transformer is skipped,
it predicts the output from cached Taylor factors (function value and
derivatives).  On steps where the transformer runs, it updates those
factors via divided differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max._core.driver import Device
from max.driver import Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType


@dataclass
class TaylorSeerState:
    """Per-request mutable state for TaylorSeer caching.

    Allocated fresh for each denoising request via
    ``TaylorSeer.create_state()``.
    """

    factor_0: Tensor | None = None
    """Cached 0th-order factor (function value)."""

    factor_1: Tensor | None = None
    """Cached 1st-order factor (first derivative approximation)."""

    factor_2: Tensor | None = None
    """Cached 2nd-order factor (second derivative approximation)."""

    last_compute_step: int | None = None
    """Step index of the last full transformer computation."""


class TaylorSeer:
    """Standalone TaylorSeer caching module.

    Compiles ``predict`` and ``update`` graphs at construction time and
    provides methods for the full TaylorSeer lifecycle: scheduling,
    prediction, factor updates, and state allocation.
    """

    def __init__(self, max_order: int, dtype: DType, device: Device) -> None:
        self.max_order = max_order
        self.dtype = dtype
        self.device = device

        self.max_order_tensor = Tensor(
            storage=Buffer.from_dlpack(
                np.array([max_order], dtype=np.int32)
            ).to(device)
        )

        tensor_type = TensorType(
            dtype, shape=["batch", "seq", "channels"], device=device
        )
        scalar_type = TensorType(DType.float32, shape=[1], device=device)
        order_type = TensorType(DType.int32, shape=[1], device=device)

        from .diffusion_pipeline import max_compile

        self.compiled_predict = max_compile(
            TaylorSeer.predict,
            input_types=[
                tensor_type,  # factor_0
                tensor_type,  # factor_1
                tensor_type,  # factor_2
                scalar_type,  # step_offset
                order_type,  # max_order
            ],
        )
        self.compiled_update = max_compile(
            TaylorSeer.update,
            input_types=[
                tensor_type,  # new_output
                tensor_type,  # old_factor_0
                tensor_type,  # old_factor_1
                scalar_type,  # delta_step
                order_type,  # max_order
            ],
        )

    @staticmethod
    def predict(
        factor_0: Tensor,
        factor_1: Tensor,
        factor_2: Tensor,
        step_offset: Tensor,
        max_order: Tensor,
    ) -> Tensor:
        """Taylor series prediction: f(t+dt) ~ f(t) + f'(t)*dt + f''(t)*dt^2/2."""
        offset = F.cast(step_offset, factor_0.dtype)
        result = factor_0 + factor_1 * offset
        offset_sq_half = (
            offset
            * offset
            * F.constant(0.5, factor_0.dtype, device=factor_0.device)
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
    def update(
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
            F.broadcast_to(use_order2, new_factor_2.shape),
            new_factor_2.dtype,
        )
        new_factor_2 = new_factor_2 * use_order2_cast

        return new_factor_0, new_factor_1, new_factor_2

    @staticmethod
    def should_skip(step: int, warmup_steps: int, cache_interval: int) -> bool:
        """Return True when the full transformer pass can be skipped at *step*."""
        if step < warmup_steps:
            return False
        return (step - warmup_steps - 1) % cache_interval != 0

    def create_state(
        self, batch_size: int, seq_len: int, output_dim: int
    ) -> TaylorSeerState:
        """Allocate fresh per-request TaylorSeer state tensors."""
        shape = (batch_size, seq_len, output_dim)

        def _zeros() -> Tensor:
            return Tensor(
                storage=Buffer.zeros(shape, self.dtype, device=self.device)
            )

        return TaylorSeerState(
            factor_0=_zeros(),
            factor_1=_zeros(),
            factor_2=_zeros(),
        )
