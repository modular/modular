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
"""CPU reference test for the V4.1 router's stable ``sqrtsoftplus``.

The routing itself runs main's router kernel (GPU only); see
``test_router_gpu.py``.
"""

from __future__ import annotations

import numpy as np
from max.pipelines.architectures.deepseekV41_modulev3.layers.router import (
    _sqrt_softplus,
)
from max.tensor import Tensor


def _sqrt_softplus_oracle(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    softplus = np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    return np.sqrt(softplus).astype(np.float32)


def test_sqrt_softplus_stable_formula() -> None:
    """The stable softplus matches the naive one on moderate logits."""
    rng = np.random.default_rng(12)
    logits = (rng.standard_normal((4, 128)) * 3.0).astype(np.float32)
    got = _sqrt_softplus(Tensor.from_dlpack(logits)).to_numpy()
    expected = np.sqrt(np.log1p(np.exp(logits))).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # Very negative logits: the naive form underflows to 0, the stable
    # form keeps the linear branch (matches the oracle's linear regime).
    extreme = np.array([[-100.0, -20.0, 0.0, 20.0, 100.0]], np.float32)
    got = _sqrt_softplus(Tensor.from_dlpack(extreme)).to_numpy()
    expected = _sqrt_softplus_oracle(extreme)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
