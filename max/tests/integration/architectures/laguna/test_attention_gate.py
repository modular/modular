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
"""The Laguna attention gate is a softplus, and softplus has a trap in it.

``log1p(exp(x))`` is the textbook form and it is wrong here: ``exp(x)``
saturates to ``inf`` around ``x = 88`` in float32, so every gate logit above
that turns the whole attention output into ``inf`` and then ``NaN``. Gate
logits that large are not hypothetical -- they are a pre-softmax linear
projection with no normalisation in front of it. This runs the helper on CPU
across both tails and pins it to ``numpy.logaddexp``.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.laguna.layers.attention import _softplus

_INPUTS = np.array(
    [-100.0, -88.0, -1.0, 0.0, 1.0, 88.0, 100.0, 1000.0], dtype=np.float32
)


def test_softplus_is_finite_and_matches_logaddexp() -> None:
    with Graph(
        "laguna_softplus",
        input_types=[
            TensorType(
                DType.float32, (_INPUTS.shape[0],), device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        (x,) = graph.inputs
        assert isinstance(x, TensorValue)
        graph.output(_softplus(x))

    session = InferenceSession(devices=[CPU()])
    (result,) = session.load(graph).execute(Buffer.from_numpy(_INPUTS))
    assert isinstance(result, Buffer)
    actual = result.to_numpy()

    assert np.isfinite(actual).all(), actual
    # Far into the negative tail the true value is subnormal (~4e-44 at -100)
    # and MAX flushes it to zero, so relative error there is meaningless while
    # the absolute error is not. Everywhere the gate can actually matter, the
    # relative tolerance is what binds.
    np.testing.assert_allclose(
        actual,
        np.logaddexp(0.0, _INPUTS),
        rtol=1e-6,
        atol=float(np.finfo(np.float32).tiny),
    )
