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

"""Graph 3b: Euler scheduler step component for Flux2Executor.

Used when TaylorSeer is enabled.  Applies the Euler update
``latents + dt * noise_pred`` as a lightweight compiled graph with
zero model weights.
"""

from __future__ import annotations

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.layer import Module
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced


class EulerStepModule(Module):
    """Euler scheduler step: ``latents + dt * noise_pred`` in float32.

    Casts to float32 for numerical stability, performs the update, then
    casts back to model dtype.  Contains no trainable weights.

    Input:  3 tensors — latents, noise_pred, dt
    Output: ``(B, seq, C)`` model dtype
    """

    def __init__(self, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device

    def __call__(
        self,
        latents: TensorValue,
        noise_pred: TensorValue,
        dt: TensorValue,
    ) -> TensorValue:
        latents_dtype = latents.dtype
        latents_f32 = ops.cast(latents, DType.float32)
        noise_pred_f32 = ops.cast(noise_pred, DType.float32)
        noise_pred_f32 = ops.rebind(noise_pred_f32, latents_f32.shape)
        updated = ops.cast(latents_f32 + dt * noise_pred_f32, latents_dtype)
        return updated

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            # latents: (B, seq, C) model dtype
            TensorType(
                self._dtype,
                shape=["batch", "seq", "channels"],
                device=self._device,
            ),
            # noise_pred: (B, seq, C) model dtype
            TensorType(
                self._dtype,
                shape=["batch", "seq", "channels"],
                device=self._device,
            ),
            # dt: (1,) float32
            TensorType(DType.float32, shape=[1], device=self._device),
        )


class DenoisePredict(CompiledComponent):
    """Graph 3b: Euler scheduler step (no model weights).

    Used when TaylorSeer is enabled.  Applies the Euler update to
    produce updated latents from ``noise_pred`` (either from the
    full transformer or from Taylor prediction).

    Output shape: ``(B, seq, C)`` model dtype.
    """

    _model: Model

    @traced(message="DenoisePredict.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        dtype: DType,
        device: Device,
    ) -> None:
        super().__init__(manifest, session)

        device_ref = DeviceRef.from_device(device)
        euler = EulerStepModule(dtype=dtype, device=device_ref)

        with Graph("denoise_predict", input_types=euler.input_types()) as graph:
            outputs = euler(*(v.tensor for v in graph.inputs))
            graph.output(outputs)

        self._model = self._load_graph(graph)

    @traced(message="DenoisePredict.__call__")
    def __call__(
        self,
        latents: Buffer,
        noise_pred: Buffer,
        dt: Buffer,
    ) -> Buffer:
        """Execute one Euler scheduler step.

        Args:
            latents: Current latent state, shape ``(B, seq, C)``.
            noise_pred: Predicted noise (from transformer or Taylor
                prediction), shape ``(B, seq, C)``.
            dt: Step delta ``sigma[i+1] - sigma[i]``, shape ``(1,)``.

        Returns:
            Updated latents, shape ``(B, seq, C)``.
        """
        result = self._model.execute(latents, noise_pred, dt)
        return result[0] if isinstance(result, (list, tuple)) else result
