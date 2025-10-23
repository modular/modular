# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import platform

import numpy as np
import pytest
import torch
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ops


@pytest.fixture(scope="module")
def hann_window_graphs() -> tuple[Graph, list[tuple[int, bool, DType]]]:
    """Create all hann window graphs once for the module."""
    graphs = {}
    device = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()

    with Graph("hann_window_all", input_types=[]) as graph:
        for window_length in [0, 1, 10, 100]:
            for periodic in [True, False]:
                for dtype in [DType.float32, DType.bfloat16]:
                    # Skip bfloat16 on ARM CPU as it's not supported
                    if dtype == DType.bfloat16 and platform.machine() in [
                        "arm64",
                        "aarch64",
                    ]:
                        continue

                    key = (window_length, periodic, dtype)
                    out = ops.hann_window(
                        window_length,
                        device=device,
                        periodic=periodic,
                        dtype=dtype,
                    )
                    graphs[key] = out.cast(DType.float32)

        # Output all graphs
        graph.output(*graphs.values())

    return graph, list(graphs.keys())


@pytest.mark.parametrize("window_length", [0, 1, 10, 100])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_hann_window(
    session: InferenceSession,
    hann_window_graphs: tuple[Graph, list[tuple[int, bool, DType]]],
    window_length: int,
    periodic: bool,
    dtype: DType,
) -> None:
    """Test hann_window against PyTorch's implementation."""
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    # Get PyTorch's implementation
    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16
    torch_window = (
        torch.hann_window(window_length, periodic=periodic, dtype=torch_dtype)
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )

    # Get our implementation using the pre-built graph
    graph, keys = hann_window_graphs

    key = (window_length, periodic, dtype)
    output_index = keys.index(key)

    model = session.load(graph)
    max_window_tensor = model()[output_index]
    assert isinstance(max_window_tensor, Tensor)
    max_window = max_window_tensor.to_numpy()

    # Compare shapes
    assert max_window.shape == torch_window.shape, (
        f"Shape mismatch: got {max_window.shape}, expected {torch_window.shape}"
    )

    # Compare values with a small tolerance
    np.testing.assert_allclose(
        max_window,
        torch_window,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Values mismatch for window_length={window_length}, periodic={periodic}",
        verbose=True,
    )
