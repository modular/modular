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
"""Shared graph and run helpers for the top-token op tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.engine import Model
from max.graph import Graph, TensorType
from max.nn._graph_ops import top_token_logit, top_token_pieces


def run_model(
    model: Model, logits: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Runs a compiled reduction on its input device.

    Args:
        model: The compiled reduction.
        logits: The input rows.

    Returns:
        The first output as a NumPy array.
    """
    buf = Buffer.from_numpy(logits).to(model.input_devices[0])
    return model(buf)[0].to_numpy()


def logit_graph(input_type: TensorType) -> Graph:
    """Builds the per-row top-token log-odds graph.

    Args:
        input_type: The shape, device and dtype of the input rows.

    Returns:
        A graph returning one score per row.
    """
    with Graph("top_token_logit", input_types=[input_type]) as graph:
        graph.output(top_token_logit(graph.inputs[0].tensor))
    return graph


def pieces_graph(input_type: TensorType, start: int) -> Graph:
    """Builds the per-row shard reduction graph.

    Args:
        input_type: The shape, device and dtype of the input rows.
        start: The shard's global vocabulary offset.

    Returns:
        A graph returning each maximum, global index and excluded mass.
    """
    with Graph("top_token_pieces", input_types=[input_type]) as graph:
        graph.output(top_token_pieces(graph.inputs[0].tensor, start))
    return graph
