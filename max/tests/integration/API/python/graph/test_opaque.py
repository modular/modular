# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine.api import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, _OpaqueType, ops


@pytest.fixture(scope="module")
def counter_ops_path() -> Path:
    return Path(os.environ["MODULAR_COUNTER_OPS_PATH"])


@pytest.fixture(scope="module")
def maker_model(session: InferenceSession, counter_ops_path: Path) -> Model:
    counter_type = _OpaqueType("Counter")
    maker_graph = Graph(
        "maker",
        input_types=[],
        output_types=[counter_type],
        custom_extensions=[counter_ops_path],
    )
    with maker_graph:
        maker_graph.output(
            ops.custom(
                "make_counter",
                device=DeviceRef.CPU(),
                values=[],
                out_types=[counter_type],
                parameters={"stride": 1},
            )[0]
        )
    maker_compiled = session.load(maker_graph)
    return maker_compiled


@pytest.fixture(scope="module")
def bumper_model(session: InferenceSession, counter_ops_path: Path) -> Model:
    counter_type = _OpaqueType("Counter")
    bumper_graph = Graph(
        "bumper",
        input_types=[counter_type],
        output_types=[],
        custom_extensions=[counter_ops_path],
    )
    with bumper_graph:
        ops.inplace_custom(
            "bump_counter",
            device=DeviceRef.CPU(),
            values=[bumper_graph.inputs[0]],
            out_types=[],
            parameters={"stride": 1},
        )
        bumper_graph.output()
    bumper_compiled = session.load(bumper_graph)
    return bumper_compiled


@pytest.fixture(scope="module")
def reader_model(session: InferenceSession, counter_ops_path: Path) -> Model:
    counter_type = _OpaqueType("Counter")
    reader_graph = Graph(
        "reader",
        input_types=[counter_type],
        output_types=[],
        custom_extensions=[counter_ops_path],
    )
    with reader_graph:
        c = ops.inplace_custom(
            "read_counter",
            device=DeviceRef.CPU(),
            values=[reader_graph.inputs[0]],
            out_types=[TensorType(DType.int32, [2], device=DeviceRef.CPU())],
            parameters={"stride": 1},
        )
        reader_graph.output(c[0])
    reader_compiled = session.load(reader_graph)
    return reader_compiled


def test_opaque_simple(
    maker_model: Model, bumper_model: Model, reader_model: Model
) -> None:
    counter = maker_model.execute()[0]
    for _ in range(5):
        bumper_model.execute(counter)
    x = reader_model.execute(counter)[0]
    assert isinstance(x, Tensor)
    result = x.to_numpy()

    assert (result == [5, 15]).all()


def test_opaque_introspection(
    maker_model: Model, bumper_model: Model, reader_model: Model
) -> None:
    assert len(maker_model.input_metadata) == 0
    assert len(maker_model.output_metadata) == 1
    assert len(bumper_model.input_metadata) == 1
    assert len(bumper_model.output_metadata) == 0
    assert len(reader_model.input_metadata) == 1
    assert len(reader_model.output_metadata) == 1


def test_opaque_type_parameterization(
    session: InferenceSession,
    custom_ops_mojopkg: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_type = TensorType(DType.int32, [12], device=DeviceRef.CPU())

    with Graph(
        "test_opaque_type_parameterization",
        input_types=[],
        custom_extensions=[custom_ops_mojopkg],
    ) as graph:
        simd_pair_type = _OpaqueType("SIMDPair", {"S0": 4, "S1": 8})

        # Create the SIMD pair using make_simd_pair custom op
        pair = ops.custom(
            "make_simd_pair",
            device=DeviceRef.CPU(),
            values=[],
            out_types=[simd_pair_type],
        )[0]

        # Process the pair with kernel_with_parameterized_opaque
        out = ops.inplace_custom(
            "kernel_with_parameterized_opaque",
            device=DeviceRef.CPU(),
            values=[pair],
            out_types=[result_type],
        )[0]

        graph.output(out)

    # Compile and execute the graph
    model = session.load(graph)
    result = model.execute()[0]

    # Verify the result
    assert isinstance(result, Tensor)
    assert result.shape == (12,)
    assert result.dtype == DType.int32

    # `make_simd_pair` performs an iota operation, validate that the values
    # in the result tensor match the expected result.
    array = result.to_numpy()
    assert np.all(array == np.arange(12, dtype=np.int32))
