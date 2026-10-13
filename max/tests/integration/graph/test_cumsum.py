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

import platform
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_cumsum(session: InferenceSession, dtype: DType) -> None:
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    input_type = TensorType(dtype, [1024], device=DeviceRef.CPU())

    with Graph(f"cumsum_{dtype}", input_types=[input_type]) as graph:
        out = ops.cumsum(graph.inputs[0].tensor, axis=0)
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16
    input_data = torch.full((1024,), 1.1, dtype=torch_dtype)

    max_result = model(
        Buffer.from_dlpack(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(max_result, Buffer)
    max_result_np = max_result.to_numpy()

    torch_result = (
        torch.cumsum(input_data, dim=0).to(dtype=torch.float32).cpu().numpy()
    )

    np.testing.assert_allclose(
        max_result_np,
        torch_result,
        rtol=1e-6,
        atol=1e-6,
        verbose=True,
    )


# (shape, axis) cases, all scanned in one graph per dtype to keep compiles few.
_GPU_CASES = [
    ((1000,), 0),
    # Rows longer than one 256-thread block, scanned per row.
    ((4, 3000), -1),
    # Scanned axis is not innermost.
    ((300, 7), 0),
    ((3, 50, 9), 1),
    # A long row among too few rows to fill the GPU is split across blocks.
    ((2, 20_011), -1),
    # A long strided axis among too few lines is split into chunks.
    ((4097, 3), 0),
]
_GPU_MODES = [(False, False), (True, False), (False, True), (True, True)]


def _check_gpu_results(
    results: Sequence[Buffer], inputs_np: Sequence[npt.NDArray[np.generic]]
) -> None:
    result_iter = iter(results)
    for input_np, (shape, axis) in zip(inputs_np, _GPU_CASES, strict=True):
        for exclusive, reverse in _GPU_MODES:
            result = next(result_iter)
            assert isinstance(result, Buffer)

            x = torch.from_numpy(input_np).to(torch.float64)
            if reverse:
                x = x.flip(axis)
            expected = torch.cumsum(x, dim=axis)
            if exclusive:
                expected = expected - x
            if reverse:
                expected = expected.flip(axis)

            np.testing.assert_array_equal(
                result.to_numpy(),
                expected.numpy().astype(input_np.dtype),
                err_msg=(
                    f"shape={shape} axis={axis} exclusive={exclusive}"
                    f" reverse={reverse}"
                ),
            )


@pytest.mark.skipif(accelerator_count() == 0, reason="requires a GPU")
@pytest.mark.parametrize("dtype", [DType.float32, DType.int32])
def test_cumsum_gpu(session: InferenceSession, dtype: DType) -> None:
    input_types = [
        TensorType(dtype, shape, device=DeviceRef.GPU())
        for shape, _ in _GPU_CASES
    ]
    # The Error policy fails graph construction if cumsum falls back to an
    # implicit host transfer.
    with Graph(
        "cumsum_gpu",
        input_types=input_types,
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        outputs = []
        for x, (_, axis) in zip(graph.inputs, _GPU_CASES, strict=True):
            for exclusive, reverse in _GPU_MODES:
                out = ops.cumsum(
                    x.tensor, axis=axis, exclusive=exclusive, reverse=reverse
                )
                assert out.device == DeviceRef.GPU()
                outputs.append(out)
        graph.output(*outputs)
    assert "mo.transfer" not in str(graph)

    model = session.load(graph)

    rng = np.random.default_rng(0)

    def random_inputs() -> list[npt.NDArray[np.generic]]:
        # Small integers keep every float32 partial sum exact.
        return [
            rng.integers(-4, 5, size=shape).astype(dtype.to_numpy())
            for shape, _ in _GPU_CASES
        ]

    inputs_np = random_inputs()
    inputs = [
        Buffer.from_numpy(arr).to(model.input_devices[0]) for arr in inputs_np
    ]
    _check_gpu_results(model(*inputs), inputs_np)

    # Capture records the launches without running them. Replay reruns every
    # one, including the split path's scratch buffers, on whatever the inputs
    # hold at replay time.
    captured = model.capture(1, *inputs)
    model.replay(1, *inputs)
    _check_gpu_results(captured, inputs_np)
    inputs_np = random_inputs()
    for buf, new_np in zip(inputs, inputs_np, strict=True):
        buf.inplace_copy_from(
            Buffer.from_numpy(new_np).to(model.input_devices[0])
        )
    model.replay(1, *inputs)
    _check_gpu_results(captured, inputs_np)
