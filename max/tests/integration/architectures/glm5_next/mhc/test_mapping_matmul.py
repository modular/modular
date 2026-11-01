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

"""The mHC mapping matmul compiles at the shape the real model runs.

``[total_seq_len, 24576] @ [24576, 24]`` in float32 is an unusual corner: K is
huge, N is tiny, M is symbolic, and float32 has no tensor-core path. Nothing
else in the tree has that combination, and it selected a split-K GEMM that
cannot handle a symbolic M. This test pins the shape so a future tiling or
heuristic change cannot silently take the mapping back there.

Compile-only and torch-free, so it is a fast gate next to the numerics in
``test_hyper_connection.py``.
"""

from __future__ import annotations

import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.glm5_next.layers.hyper_connection import (
    HyperConnection,
)

HIDDEN_SIZE = 4096
HC_MULT = 4


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator(0)])


@pytest.mark.parametrize(
    "tokens",
    [
        pytest.param("total_seq_len", id="symbolic_m"),
        pytest.param(17, id="static_m"),
    ],
)
def test_mapping_compiles(session: InferenceSession, tokens: str | int) -> None:
    """The site compiles with a symbolic token count, as the model builds it.

    A static M is the easy case and is what the numerics test uses; the
    symbolic M is the one that broke, so both are pinned.
    """
    site = HyperConnection(
        hidden_size=HIDDEN_SIZE,
        hc_mult=HC_MULT,
        dtype=DType.bfloat16,
        name="hc_attn",
    )

    def mapping(streams: TensorValue) -> tuple[TensorValue, ...]:
        # The site returns a `StreamMixing` dataclass, which `Graph` cannot
        # take as its outputs; unpack it into the value sequence it wants.
        mixing = site(streams)
        return mixing.post, mixing.comb, mixing.xs

    graph = Graph(
        "Glm5NextHyperConnectionMapping",
        mapping,
        input_types=(
            TensorType(
                DType.bfloat16,
                (tokens, HC_MULT, HIDDEN_SIZE),
                device=DeviceRef.GPU(),
            ),
        ),
    )
    session.load(graph, weights_registry=site.state_dict())
