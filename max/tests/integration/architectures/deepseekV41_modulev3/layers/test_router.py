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


def test_sqrt_softplus_stable_formula() -> None:
    """The stable softplus matches the naive one on moderate logits."""
    rng = np.random.default_rng(12)
    logits = (rng.standard_normal((4, 128)) * 3.0).astype(np.float32)
    got = _sqrt_softplus(Tensor.from_dlpack(logits)).to_numpy()
    expected = np.sqrt(np.log1p(np.exp(logits))).astype(np.float32)
    # The ``log(1 + exp(-|x|))`` form rounds ``1 + exp(-|x|)`` to fp32, so
    # a logit near -10 moves its score by about 1e-6 (see the log1p test).
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-5)
    # Check the stable branch where the direct exp overflows or underflows.
    extreme = np.array([[-100.0, -20.0, 0.0, 20.0, 100.0]], np.float32)
    got = _sqrt_softplus(Tensor.from_dlpack(extreme)).to_numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(
        got[0, 2:], [np.sqrt(np.log(2)), np.sqrt(20), 10]
    )


def test_sqrt_softplus_matches_the_log1p_form_within_bound() -> None:
    logits = np.linspace(-40.0, 40.0, 4001, dtype=np.float32)[None, :]
    got = _sqrt_softplus(Tensor.from_dlpack(logits)).to_numpy()
    log1p_form = np.sqrt(
        np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    ).astype(np.float32)
    np.testing.assert_allclose(got, log1p_form, rtol=1e-6, atol=3e-4)
    positive = logits >= 0.0
    np.testing.assert_allclose(
        got[positive], log1p_form[positive], rtol=1e-5, atol=0
    )
