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

"""Classifier-free guidance combine as a standalone compiled graph.

Computes ``neg + guidance_scale * (pos - neg)`` in float32 and casts the
result back to the input dtype. Matches the V3 Klein pipeline's
``cfg_combine`` (``pipeline_flux2_klein.py:106-121``) exactly.
"""

from __future__ import annotations

from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.layer import Module
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced


class CfgCombineModule(Module):
    """Classifier-free guidance blend in float32."""

    def __init__(self, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device

    def __call__(
        self,
        pos_noise: TensorValue,
        neg_noise: TensorValue,
        guidance_scale: TensorValue,
    ) -> TensorValue:
        input_dtype = pos_noise.dtype
        pos_f32 = ops.cast(pos_noise, DType.float32)
        neg_f32 = ops.cast(neg_noise, DType.float32)
        combined = neg_f32 + guidance_scale * (pos_f32 - neg_f32)
        return ops.cast(combined, input_dtype)

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch", "seq", "channels"],
                device=self._device,
            ),
            TensorType(
                self._dtype,
                shape=["batch", "seq", "channels"],
                device=self._device,
            ),
            TensorType(DType.float32, shape=[], device=self._device),
        )


class CfgCombineComponent(CompiledComponent):
    """Compiled CFG combine graph."""

    _model: Model

    @traced(message="CfgCombineComponent.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__(manifest, session)

        module = CfgCombineModule(dtype=dtype, device=device)
        with Graph("cfg_combine", input_types=module.input_types()) as graph:
            outputs = module(*(v.tensor for v in graph.inputs))
            graph.output(outputs)

        self._model = self._load_graph(graph)

    @traced(message="CfgCombineComponent.__call__")
    def __call__(
        self,
        pos_noise: Buffer,
        neg_noise: Buffer,
        guidance_scale: Buffer,
    ) -> Buffer:
        """Blend positive and negative noise predictions via CFG."""
        result = self._model.execute(pos_noise, neg_noise, guidance_scale)
        return result[0] if isinstance(result, (list, tuple)) else result
