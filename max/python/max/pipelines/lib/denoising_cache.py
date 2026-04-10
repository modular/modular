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

"""Buffer-based TaylorSeer cache for executor-style diffusion pipelines.

This module provides :class:`TaylorSeerCache`, a high-level abstraction
that compiles Taylor series predict/update graphs through a shared
:class:`InferenceSession` and operates entirely on :class:`Buffer` objects
(driver API).  This is the executor-path counterpart to
:class:`~max.pipelines.lib.interfaces.taylorseer.TaylorSeer`, which
creates its own private session and uses eager :class:`Tensor` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph

from .interfaces.cache_mixin import DenoisingCacheConfig
from .interfaces.taylorseer import (
    TaylorSeer,
    _TaylorPredictModule,
    _TaylorUpdateModule,
)


@dataclass
class TaylorSeerBufferState:
    """Per-request mutable TaylorSeer state using Buffer objects.

    Allocated fresh for each denoising request via
    :meth:`TaylorSeerCache.create_state`.
    """

    factor_0: Buffer
    """Cached 0th-order factor (function value)."""

    factor_1: Buffer
    """Cached 1st-order factor (first derivative approximation)."""

    factor_2: Buffer
    """Cached 2nd-order factor (second derivative approximation)."""

    last_compute_step: int | None = None
    """Step index of the last full transformer computation."""


class TaylorSeerCache:
    """High-level TaylorSeer for executor pipelines (Buffer-based).

    Compiles predict and update graphs through the executor's shared
    :class:`InferenceSession` at construction time.  All runtime methods
    accept and return :class:`Buffer` objects, matching the executor's
    driver-level API.

    Args:
        config: Denoising cache configuration (must have ``taylorseer=True``
            and resolved non-None fields for interval/warmup/order).
        dtype: Model compute dtype (e.g. ``DType.bfloat16``).
        device: Target device for graph execution.
        session: The executor's shared inference session.
    """

    def __init__(
        self,
        config: DenoisingCacheConfig,
        dtype: DType,
        device: Device,
        session: InferenceSession,
    ) -> None:
        self._dtype = dtype
        self._device = device

        assert config.taylorseer_cache_interval is not None
        assert config.taylorseer_warmup_steps is not None
        assert config.taylorseer_max_order is not None

        self._cache_interval: int = config.taylorseer_cache_interval
        self._warmup_steps: int = config.taylorseer_warmup_steps
        self._max_order: int = config.taylorseer_max_order

        # Pre-allocate max_order scalar on device.
        self._max_order_buf: Buffer = Buffer.from_dlpack(
            np.array([self._max_order], dtype=np.int32)
        ).to(device)

        # Compile predict graph.
        device_ref = DeviceRef.from_device(device)
        predict_module = _TaylorPredictModule(dtype, device_ref)
        with Graph(
            "taylor_predict", input_types=predict_module.input_types()
        ) as predict_graph:
            predict_output = predict_module(
                *(value.tensor for value in predict_graph.inputs)
            )
            predict_graph.output(predict_output)
        self._predict_model: Model = session.load(predict_graph)

        # Compile update graph.
        update_module = _TaylorUpdateModule(dtype, device_ref)
        with Graph(
            "taylor_update", input_types=update_module.input_types()
        ) as update_graph:
            update_outputs = update_module(
                *(value.tensor for value in update_graph.inputs)
            )
            update_graph.output(*update_outputs)
        self._update_model: Model = session.load(update_graph)

    def create_state(
        self, batch_size: int, seq_len: int, output_dim: int
    ) -> TaylorSeerBufferState:
        """Allocate fresh per-request TaylorSeer state buffers.

        Args:
            batch_size: Batch dimension.
            seq_len: Sequence length (packed latent tokens).
            output_dim: Channel dimension of noise_pred.

        Returns:
            A new :class:`TaylorSeerBufferState` with zero-initialized
            factor buffers on the target device.
        """
        shape = (batch_size, seq_len, output_dim)
        return TaylorSeerBufferState(
            factor_0=Buffer.zeros(shape, self._dtype, device=self._device),
            factor_1=Buffer.zeros(shape, self._dtype, device=self._device),
            factor_2=Buffer.zeros(shape, self._dtype, device=self._device),
        )

    def should_skip(self, step: int) -> bool:
        """Return True when the full transformer pass can be skipped."""
        return TaylorSeer.should_skip(
            step, self._warmup_steps, self._cache_interval
        )

    def predict(self, state: TaylorSeerBufferState, step: int) -> Buffer:
        """Predict noise_pred from cached Taylor factors.

        Args:
            state: Current per-request TaylorSeer state.
            step: Current denoising step index.

        Returns:
            Predicted noise_pred buffer, shape ``(B, seq, C)``.
        """
        delta = (
            float(step - state.last_compute_step)
            if state.last_compute_step is not None
            else 1.0
        )
        step_offset = Buffer.from_dlpack(
            np.array([delta], dtype=np.float32)
        ).to(self._device)

        result = self._predict_model.execute(
            state.factor_0,
            state.factor_1,
            state.factor_2,
            step_offset,
            self._max_order_buf,
        )
        return result[0] if isinstance(result, (list, tuple)) else result

    def update(
        self,
        state: TaylorSeerBufferState,
        noise_pred: Buffer,
        step: int,
    ) -> None:
        """Update Taylor factors from a full transformer computation.

        Mutates *state* in-place with new factor values.

        Args:
            state: Current per-request TaylorSeer state.
            noise_pred: Fresh noise_pred from the transformer, shape
                ``(B, seq, C)``.
            step: Current denoising step index.
        """
        delta = (
            float(step - state.last_compute_step)
            if state.last_compute_step is not None
            else 1.0
        )
        delta_step = Buffer.from_dlpack(np.array([delta], dtype=np.float32)).to(
            self._device
        )

        results = self._update_model.execute(
            noise_pred,
            state.factor_0,
            state.factor_1,
            delta_step,
            self._max_order_buf,
        )
        state.factor_0 = results[0]
        state.factor_1 = results[1]
        state.factor_2 = results[2]
        state.last_compute_step = step
