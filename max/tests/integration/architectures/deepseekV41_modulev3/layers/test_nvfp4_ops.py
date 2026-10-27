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
"""CPU reference tests for the V4.1 expert SwiGLU clamp.

The asymmetric training clamp gates only the *upper* side; ``up`` clamps
both. The oracle is NumPy: clamp, then ``silu(gate) * up`` in fp32.
"""

from __future__ import annotations

import numpy as np
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV41_modulev3.layers.nvfp4_ops import (
    swiglu,
)


def _oracle(
    gate: np.ndarray, up: np.ndarray, limit: float | None
) -> np.ndarray:
    gate = gate.astype(np.float32)
    up = up.astype(np.float32)
    if limit is not None:
        gate = np.minimum(gate, limit)
        up = np.clip(up, -limit, limit)
    return (gate / (1.0 + np.exp(-gate)) * up).astype(np.float32)


def test_asymmetric_clamp_production_width() -> None:
    """V4.1's 1536-wide expert intermediate with limit 10."""
    rng = np.random.default_rng(16)
    gate = (rng.standard_normal((5, 1536)) * 6.0).astype(np.float32)
    up = (rng.standard_normal((5, 1536)) * 6.0).astype(np.float32)
    got = swiglu(Tensor.from_dlpack(gate), Tensor.from_dlpack(up), 10.0)
    out = got.to_numpy()
    assert out.dtype == np.float32
    np.testing.assert_allclose(
        out, _oracle(gate, up, 10.0), rtol=1e-6, atol=1e-6
    )
    # The clamp is asymmetric: a gate of -15 keeps its value (silu ~ -1e-5),
    # an up of -15 clamps to -10.
    gate = np.array([[15.0, -15.0, 0.5, 11.0]], np.float32)
    up = np.array([[15.0, -15.0, -0.5, 11.0]], np.float32)
    out = swiglu(
        Tensor.from_dlpack(gate), Tensor.from_dlpack(up), 10.0
    ).to_numpy()
    np.testing.assert_allclose(
        out, _oracle(gate, up, 10.0), rtol=1e-6, atol=1e-6
    )
    assert out[0, 1] > 1e-5  # gate -15 survives: silu(-15) * (-10) > 0


def test_no_limit_disables_clamp() -> None:
    rng = np.random.default_rng(17)
    gate = (rng.standard_normal((3, 256)) * 30.0).astype(np.float32)
    up = (rng.standard_normal((3, 256)) * 30.0).astype(np.float32)
    out = swiglu(
        Tensor.from_dlpack(gate), Tensor.from_dlpack(up), None
    ).to_numpy()
    np.testing.assert_allclose(
        out, _oracle(gate, up, None), rtol=1e-6, atol=1e-6
    )
