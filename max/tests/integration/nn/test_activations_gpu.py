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
"""Numerical correctness tests for activation functions via the
experimental tensor API.

Mirrors the kernel-path probe in ``oss/max-llm-book/tests/test_gpt2_accel.py``:
construct a tensor, run ``F.<activation>(...)`` under ``default_device(...)``,
copy the result back to CPU, and compare against the torch reference.

Parametrized over available backends. CPU is always exercised; the
``Accelerator()`` case is added only when ``accelerator_count() > 0`` so the
file remains importable on CPU-only hosts.
"""

import max.experimental.functional as F
import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.dtype import DType
from max.experimental.tensor import Tensor, default_device, default_dtype

_DEVICE_PARAMS = [pytest.param(CPU, id="cpu")]
if accelerator_count() > 0:
    _DEVICE_PARAMS.append(pytest.param(Accelerator, id="accel"))


@pytest.fixture(params=_DEVICE_PARAMS)
def device(request: pytest.FixtureRequest) -> Device:
    # Instantiate per-test so a fresh device handle is used.
    return request.param()


def _torch_gelu(x: torch.Tensor, approximate: str) -> torch.Tensor:
    if approximate in ("none", "tanh"):
        return torch.nn.functional.gelu(x, approximate=approximate)
    # "quick" GELU is not a torch built-in: x * sigmoid(1.702 * x).
    return x * torch.sigmoid(1.702 * x)


@pytest.mark.parametrize("approximate", ["none", "tanh", "quick"])
def test_gelu_matches_torch(device: Device, approximate: str) -> None:
    # Span the activation domain: near-zero, the curvy region, and both tails.
    x_np = np.linspace(-4.0, 4.0, 32, dtype=np.float32).reshape(4, 8)

    with default_device(device), default_dtype(DType.float32):
        x = Tensor.from_dlpack(x_np).to(device)
        out = F.gelu(x, approximate=approximate)
        max_result = np.from_dlpack(out.to(CPU()))

    torch_result = _torch_gelu(torch.from_numpy(x_np), approximate).numpy()

    # "quick" is itself an approximation of GELU; we test the kernel
    # against the same formula, but allow looser tolerance for fp32 drift.
    atol = 1e-3 if approximate == "quick" else 1e-5
    rtol = 1e-3 if approximate == "quick" else 1e-5
    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=rtol,
        atol=atol,
    )
