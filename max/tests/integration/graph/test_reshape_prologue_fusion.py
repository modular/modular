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

"""End-to-end test for reshape-into-prologue fusion under advanced fusion.

The load-side mirror of ``test_reshape_epilogue_fusion.py``: a reshape feeding
a fused input folds into the consumer's prologue as a ``mogg.index.reshape``
load-index transform, so the load reads from the reshaped position and the
reshape no longer materializes its own kernel.

The case here is the one that needs the runtime shapes. An INNER dynamic dim
re-linearizes across itself, so the transform cannot read the shapes from the
static signature and instead reads the pair threaded into the prologue functor
as shape captures. Element values are distinct so a scrambled load index fails
the comparison rather than merely producing a wrong shape.
"""

from __future__ import annotations

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_reshape_fuses_into_gather_data_inner_dynamic(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """``gather(reshape(x, [2, D]), indices)`` with an INNER dynamic dim ``D``.

    ``[D, 2] -> [2, D]`` re-linearizes across ``D``, so the load-index
    transform reads the runtime from/to shapes carried into the prologue.
    """
    with Graph(
        "reshape_fuses_into_gather_data_inner_dynamic",
        input_types=[
            TensorType(DType.float32, ["D", 2], device=DeviceRef.CPU()),
            TensorType(DType.int32, [2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, indices = (v.tensor for v in graph.inputs)
        data = ops.reshape(x, [2, x.shape[0]])
        graph.output(ops.gather(data, indices, axis=0))

    x_np = np.arange(8, dtype=np.float32).reshape(4, 2)
    idx_np = np.array([1, 0], dtype=np.int32)
    (out,) = run_and_verify_fusion(
        session,
        graph,
        x_np,
        idx_np,
        fused=r"mo\.static\.reshape.*mo\.gather",
    )
    np.testing.assert_allclose(
        out, x_np.reshape(2, 4)[idx_np], rtol=1e-5, atol=1e-5
    )
