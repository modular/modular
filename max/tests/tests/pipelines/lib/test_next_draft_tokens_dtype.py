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
"""The overlap pipeline takes the drafts at the dtype the graph declares.

Most spec-decode graphs emit int64 ``next_draft_tokens``, but DeepSeek-V4.1's
ModuleV3 DSpark graph emits int32 ones, so the pinned buffer the drafts are
copied into and the next step's realize graph follow the graph's declared
dtype, widening to the int64 draft tokens the forward takes.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _next_draft_tokens_dtype,
    build_realize_future_token_graph,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
)

K = 3


@pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
def test_the_drafts_dtype_is_the_graphs_declared_output(dtype: DType) -> None:
    model = object.__new__(SpecDecodePipelineModel)
    model.model = cast(
        Any,
        SimpleNamespace(
            output_metadata=[
                SimpleNamespace(dtype=DType.int64),
                SimpleNamespace(dtype=DType.int64),
                SimpleNamespace(dtype=dtype),
            ]
        ),
    )
    assert _next_draft_tokens_dtype(model) == dtype


def test_a_model_without_a_spec_graph_keeps_int64_drafts() -> None:
    assert _next_draft_tokens_dtype(object()) == DType.int64


@pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
def test_the_realize_graph_widens_the_previous_drafts(dtype: DType) -> None:
    graph = build_realize_future_token_graph(
        devices=[DeviceRef.CPU()],
        enable_dp=False,
        num_speculative_tokens=K,
        next_draft_tokens_dtype=dtype,
    )
    prev_drafts = [
        value.type
        for value in graph.inputs
        if isinstance(value.type, TensorType)
        and "gen_num_draft_tokens" in [str(d) for d in value.type.shape]
    ]
    assert [t.dtype for t in prev_drafts] == [dtype]
    realized_drafts = graph.output_types[1]
    assert isinstance(realized_drafts, TensorType)
    assert realized_drafts.dtype == DType.int64
