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
"""Tests the attention gate's softplus.

``log(1 + exp(x))`` overflows to ``inf`` well inside the range the gate sees,
so the layer folds out ``max(x, 0)``. These cases pin that the folded form
agrees with the direct one where both are finite, and stays finite where the
direct one is not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.laguna_modulev3.layers.attention import (
    softplus,
)


class _Softplus(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return softplus(x)


def _softplus(values: list[float]) -> np.ndarray:
    with F.lazy(), default_dtype(DType.float32):
        layer = _Softplus()
        layer.to(CPU())
    compiled = layer.compile(
        TensorType(DType.float32, (len(values),), device=DeviceRef.CPU())
    )
    (buf,) = compiled.execute_raw(
        Buffer.from_dlpack(torch.tensor(values, dtype=torch.float32))
    )
    return torch.from_dlpack(buf).float().numpy()


@pytest.mark.parametrize("values", [[-8.0, -1.0, 0.0, 1.0, 8.0], [-30.0, 30.0]])
def test_softplus_matches_the_direct_form(values: list[float]) -> None:
    expected = torch.nn.functional.softplus(
        torch.tensor(values, dtype=torch.float32)
    ).numpy()
    np.testing.assert_allclose(
        _softplus(values), expected, rtol=1e-5, atol=1e-5
    )


def test_softplus_is_finite_where_the_direct_form_overflows() -> None:
    """``log1p(exp(100))`` is ``inf`` in float32; the gate needs a value."""
    assert not np.isfinite(np.log1p(np.exp(np.float32(100.0))))
    got = _softplus([100.0])
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, [100.0], rtol=1e-5, atol=1e-5)
