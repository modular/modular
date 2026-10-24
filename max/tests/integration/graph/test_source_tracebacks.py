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
"""Source tracebacks show the user's frames, and execute errors have a type."""

from collections.abc import Iterator

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, ModelExecutionError
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.fixture
def source_tracebacks() -> Iterator[None]:
    debug = InferenceSession.debug
    saved = (debug.source_tracebacks, debug.source_tracebacks_hide_internal)
    debug.source_tracebacks = True
    yield
    debug.source_tracebacks, debug.source_tracebacks_hide_internal = saved


def _compile_error_traceback() -> str:
    # Rank 9 exceeds what the graph compiler lowers, so compiling fails with
    # the op's source traceback.
    tensor_type = TensorType(DType.float32, [1] * 9, device=DeviceRef.CPU())
    with Graph("rank9", input_types=(tensor_type, tensor_type)) as graph:
        a, b = (value.tensor for value in graph.inputs)
        graph.output(ops.add(a, b))
    with pytest.raises(RuntimeError) as err:
        InferenceSession().load(graph)
    messages = []
    exc: BaseException | None = err.value
    while exc is not None:
        messages.append(str(exc))
        exc = exc.__cause__
    return "\n".join(messages)


@pytest.mark.compile_on_gpu("asserts on the error its compile raises")
@pytest.mark.usefixtures("source_tracebacks")
def test_tracebacks_hide_max_frames_by_default() -> None:
    message = _compile_error_traceback()
    assert "Source Traceback:" in message
    assert "test_source_tracebacks.py" in message
    assert "max/graph/" not in message


@pytest.mark.compile_on_gpu("asserts on the error its compile raises")
@pytest.mark.usefixtures("source_tracebacks")
def test_tracebacks_keep_max_frames_on_request() -> None:
    InferenceSession.debug.source_tracebacks_hide_internal = False
    message = _compile_error_traceback()
    assert "test_source_tracebacks.py" in message
    assert "max/graph/" in message


def test_execute_error_is_kernel_execution_error() -> None:
    a_type = TensorType(DType.float32, ["N", 4], device=DeviceRef.CPU())
    with Graph("dim_mismatch", input_types=(a_type, a_type)) as graph:
        a, b = (value.tensor for value in graph.inputs)
        graph.output(ops.add(a, b))
    model = InferenceSession().load(graph)

    with pytest.raises(ModelExecutionError) as err:
        model.execute(
            Buffer.from_numpy(np.ones((2, 4), dtype=np.float32)),
            Buffer.from_numpy(np.ones((3, 4), dtype=np.float32)),
        )
    # A RuntimeError, and still a ValueError for older handlers.
    assert isinstance(err.value, RuntimeError)
    assert isinstance(err.value, ValueError)
    assert "symbolic dimension 'N'" in str(err.value)
