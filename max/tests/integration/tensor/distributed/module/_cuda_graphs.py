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
"""Shared test logic for CUDA graph capture/replay with CompiledModel.

DO NOT run this file directly -- it contains a base class that is
subclassed by module/test_cuda_graphs_multi_gpu.py.

These tests exercise the pipeline-builder workflow:
compile → execute_raw → capture → replay, verifying that CompiledModel
exposes the right tools for the serving infrastructure.

Single-GPU only — multi-device collectives are not CUDA-graph-capture-safe,
so multi-GPU capture/replay is out of scope here.

Subclasses must define:
    DEVICE: Device -- a single GPU device
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import Model
from max.experimental.distributed_functional import full, matmul
from max.experimental.nn.module import CompiledModel, Module, module_dataclass
from max.experimental.tensor import Tensor, TensorType
from max.graph import DeviceRef

D = 4
F32 = DType.float32


class CUDAGraphTests:
    """Tests for CUDA graph capture/replay via CompiledModel.

    Subclass must set DEVICE.
    """

    DEVICE: ClassVar[Device]

    def _single_gpu_model(self) -> CompiledModel:
        """Compiles a simple linear model on a single GPU."""
        device = self.DEVICE
        device_ref = DeviceRef(device.label, device.id)

        @module_dataclass
        class Linear(Module[[Tensor], Tensor]):
            W: Tensor

            def forward(self, x: Tensor) -> Tensor:
                return matmul(x, self.W)

        W = full([D, D], 1.0, dtype=F32, device=device)
        model = Linear(W=W)
        model.to(device)
        input_type = TensorType(F32, ["batch", D], device=device_ref)
        return model.compile(input_type)

    def _gpu_buf(self, arr: np.ndarray) -> Buffer:
        return Buffer.from_numpy(arr).to(self.DEVICE)

    def test_execute_raw(self) -> None:
        """execute_raw returns flat Buffers with correct values."""
        compiled = self._single_gpu_model()
        x = self._gpu_buf(np.ones((3, D), dtype=np.float32))
        raw = compiled.execute_raw(x)
        assert len(raw) == 1
        assert isinstance(raw[0], Buffer)
        result = Tensor(storage=raw[0]).to_numpy()
        np.testing.assert_allclose(result, np.full((3, D), D), rtol=1e-4)

    def test_execute_raw_matches_tensor_path(self) -> None:
        """execute_raw and __call__ produce the same values."""
        compiled = self._single_gpu_model()

        x_buf = self._gpu_buf(np.ones((3, D), dtype=np.float32))
        raw = compiled.execute_raw(x_buf)

        x_tensor = full([3, D], 1.0, dtype=F32, device=self.DEVICE)
        tensor_result = compiled(x_tensor)

        np.testing.assert_allclose(
            Tensor(storage=raw[0]).to_numpy(),
            tensor_result.to_numpy(),
            rtol=1e-4,
        )

    def test_capture_and_replay(self) -> None:
        """Capture a CUDA graph, replay with new data, verify outputs."""
        compiled = self._single_gpu_model()
        engine = compiled.engine_model

        x = self._gpu_buf(np.ones((3, D), dtype=np.float32))

        # Capture: records the kernel launch sequence.
        graph_key = 1
        capture_outputs = engine.capture(graph_key, x)
        assert len(capture_outputs) == 1

        # Replay with 2s instead of 1s — copy into the *same* buffer.
        x.inplace_copy_from(
            self._gpu_buf(np.full((3, D), 2.0, dtype=np.float32))
        )
        engine.replay(graph_key, x)
        # Outputs are updated in-place in the captured output buffers.
        result = Tensor(storage=capture_outputs[0]).to_numpy()
        # 2s @ ones_weight = 2*D per element
        np.testing.assert_allclose(result, np.full((3, D), 2 * D), rtol=1e-4)

    def test_capture_multiple_batch_sizes(self) -> None:
        """Capture separate graphs for different batch sizes, like serving."""
        compiled = self._single_gpu_model()
        engine = compiled.engine_model

        entries: dict[int, tuple[Buffer, list[Buffer]]] = {}
        for batch_size in [1, 2, 4]:
            x = self._gpu_buf(np.ones((batch_size, D), dtype=np.float32))
            outputs = engine.capture(batch_size, x)
            entries[batch_size] = (x, outputs)

        # Replay each with different data — mimics serving decode loop.
        for batch_size, (captured_input, captured_outputs) in entries.items():
            captured_input.inplace_copy_from(
                self._gpu_buf(np.full((batch_size, D), 3.0, dtype=np.float32))
            )
            engine.replay(batch_size, captured_input)
            result = Tensor(storage=captured_outputs[0]).to_numpy()
            np.testing.assert_allclose(
                result, np.full((batch_size, D), 3 * D), rtol=1e-4
            )

    def test_debug_verify_replay(self) -> None:
        """debug_verify_replay does not raise after a valid capture."""
        compiled = self._single_gpu_model()
        engine = compiled.engine_model

        x = self._gpu_buf(np.ones((2, D), dtype=np.float32))
        engine.capture(42, x)
        # Should not raise — eager trace matches the captured graph.
        engine.debug_verify_replay(42, x)

    def test_signal_buffers_empty_single_gpu(self) -> None:
        """Single-GPU model has no signal buffers."""
        compiled = self._single_gpu_model()
        assert compiled.signal_buffers == []

    def test_pipeline_builder_pattern(self) -> None:
        """End-to-end pipeline builder pattern: compile, expose engine_model,
        execute_raw for normal path, capture/replay for decode.

        This mimics what a V3 pipeline model (e.g. llama3_modulev3) would do.
        """
        compiled = self._single_gpu_model()

        # Pipeline builder wires up model attribute for _SupportsModelCapture.
        model = compiled.engine_model
        assert isinstance(model, Model)

        # Normal execution path (no CUDA graphs) — pipeline calls execute_raw.
        x = self._gpu_buf(np.ones((3, D), dtype=np.float32))
        raw_outputs = compiled.execute_raw(x)
        assert len(raw_outputs) == 1
        eager_result = Tensor(storage=raw_outputs[0]).to_numpy()

        # Warmup: capture graphs for batch sizes 1..3 (like ServeGraphCaptureRunner).
        captured: dict[int, tuple[Buffer, list[Buffer]]] = {}
        for bs in [1, 2, 3]:
            buf = self._gpu_buf(np.ones((bs, D), dtype=np.float32))
            outputs = model.capture(bs, buf)
            captured[bs] = (buf, outputs)

        # Serving loop: replay for batch_size=3 with new data.
        cap_input, cap_outputs = captured[3]
        new_x = self._gpu_buf(np.full((3, D), 5.0, dtype=np.float32))
        cap_input.inplace_copy_from(new_x)
        model.replay(3, cap_input)
        replay_result = Tensor(storage=cap_outputs[0]).to_numpy()
        # 5s @ ones_weight = 5*D per element
        np.testing.assert_allclose(
            replay_result, np.full((3, D), 5 * D), rtol=1e-4
        )

        # Fallback to eager when batch_size exceeds captured range.
        x_large = self._gpu_buf(np.full((8, D), 2.0, dtype=np.float32))
        fallback = compiled.execute_raw(x_large)
        fallback_result = Tensor(storage=fallback[0]).to_numpy()
        np.testing.assert_allclose(
            fallback_result, np.full((8, D), 2 * D), rtol=1e-4
        )
