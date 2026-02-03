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
"""GPU tests for MO interpreter constant handling."""

import numpy as np
import pytest
import torch
from max._interpreter import MOInterpreter
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import Graph, ops


class TestConstantGPU:
    """Tests for GPU constant handling in the interpreter."""

    def test_constant_on_gpu(self) -> None:
        """Test that constants can be created directly on GPU."""
        gpu = Accelerator()

        with Graph("gpu_constant", input_types=[]) as graph:
            c = ops.constant([1.0, 2.0, 3.0], dtype=DType.float32, device=gpu)
            graph.output(c)

        interp = MOInterpreter(devices=[CPU(), gpu])
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        assert not result.device.is_host

        # Verify values using PyTorch
        torch_tensor = torch.from_dlpack(result)
        expected = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(torch_tensor, expected)

    def test_constant_gpu_2d(self) -> None:
        """Test 2D constant on GPU."""
        gpu = Accelerator()

        with Graph("gpu_constant_2d", input_types=[]) as graph:
            c = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=DType.float32,
                device=gpu,
            )
            graph.output(c)

        interp = MOInterpreter(devices=[CPU(), gpu])
        outputs = interp.execute(graph, [])

        result = outputs[0]
        assert isinstance(result, Buffer)
        assert not result.device.is_host

        # Verify via to_numpy (GPU->CPU transfer)
        np.testing.assert_array_almost_equal(
            result.to_numpy(),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

    def test_constant_mixed_devices(self) -> None:
        """Test graph with constants on both CPU and GPU."""
        cpu = CPU()
        gpu = Accelerator()

        with Graph("mixed_constants", input_types=[]) as graph:
            cpu_const = ops.constant(
                [1.0, 2.0], dtype=DType.float32, device=cpu
            )
            gpu_const = ops.constant(
                [3.0, 4.0], dtype=DType.float32, device=gpu
            )
            graph.output(cpu_const, gpu_const)

        interp = MOInterpreter(devices=[cpu, gpu])
        outputs = interp.execute(graph, [])

        assert len(outputs) == 2
        assert isinstance(outputs[0], Buffer)
        assert isinstance(outputs[1], Buffer)
        assert outputs[0].device.is_host
        assert not outputs[1].device.is_host

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float64, DType.int32]
    )
    def test_constant_gpu_dtypes(self, dtype: DType) -> None:
        """Test GPU constants with various dtypes."""
        gpu = Accelerator()
        np_dtype = dtype.to_numpy()

        with Graph(f"gpu_constant_{dtype}", input_types=[]) as graph:
            c = ops.constant(np.array([1, 2, 3, 4], dtype=np_dtype), device=gpu)
            graph.output(c)

        interp = MOInterpreter(devices=[CPU(), gpu])
        outputs = interp.execute(graph, [])

        result = outputs[0]
        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == dtype
