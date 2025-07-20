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
"""Test loading external safetensor weights into Max Graph."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import SafetensorWeights


def test_load_safetensors_one(testdata_directory) -> None:  # noqa: ANN001
    weights = SafetensorWeights(
        [testdata_directory / "example_data_1.safetensors"]
    )
    with Graph("test_load_safetensors1") as graph:
        data = {
            key: graph.add_weight(weight.allocate())
            for key, weight in weights.items()
        }
        assert len(data) == 7
        assert data["1.a"].type == TensorType(
            DType.int32, [5, 2], device=DeviceRef.CPU()
        )
        assert data["1.b"].type == TensorType(
            DType.float64, [1, 2, 3], device=DeviceRef.CPU()
        )
        assert data["1.c"].type == TensorType(
            DType.float32, [], device=DeviceRef.CPU()
        )
        assert data["1.fancy/name"].type == TensorType(
            DType.int64, [3], device=DeviceRef.CPU()
        )
        assert data["1.bf16"].type == TensorType(
            DType.bfloat16, [2], device=DeviceRef.CPU()
        )
        assert data["1.float8_e4m3fn"].type == TensorType(
            DType.float8_e4m3fn, [2], device=DeviceRef.CPU()
        )
        assert data["1.float8_e5m2"].type == TensorType(
            DType.float8_e5m2, [2], device=DeviceRef.CPU()
        )


def test_load_safetensors_multi(testdata_directory) -> None:  # noqa: ANN001
    weights = SafetensorWeights(
        [
            testdata_directory / "example_data_1.safetensors",
            testdata_directory / "example_data_2.safetensors",
        ]
    )
    with Graph("test_load_safetensors_multi") as graph:
        data = {
            key: graph.add_weight(weight.allocate())
            for key, weight in weights.items()
        }
        assert len(data) == 14
        assert data["1.a"].type == TensorType(
            DType.int32, [5, 2], device=DeviceRef.CPU()
        )
        assert data["1.b"].type == TensorType(
            DType.float64, [1, 2, 3], device=DeviceRef.CPU()
        )
        assert data["1.c"].type == TensorType(
            DType.float32, [], device=DeviceRef.CPU()
        )
        assert data["1.fancy/name"].type == TensorType(
            DType.int64, [3], device=DeviceRef.CPU()
        )
        assert data["1.bf16"].type == TensorType(
            DType.bfloat16, [2], device=DeviceRef.CPU()
        )
        assert data["1.float8_e4m3fn"].type == TensorType(
            DType.float8_e4m3fn, [2], device=DeviceRef.CPU()
        )
        assert data["1.float8_e5m2"].type == TensorType(
            DType.float8_e5m2, [2], device=DeviceRef.CPU()
        )
        assert data["2.a"].type == TensorType(
            DType.int32, [5, 2], device=DeviceRef.CPU()
        )
        assert data["2.b"].type == TensorType(
            DType.float64, [1, 2, 3], device=DeviceRef.CPU()
        )
        assert data["2.c"].type == TensorType(
            DType.float32, [], device=DeviceRef.CPU()
        )
        assert data["2.fancy/name"].type == TensorType(
            DType.int64, [3], device=DeviceRef.CPU()
        )
        assert data["2.bf16"].type == TensorType(
            DType.bfloat16, [2], device=DeviceRef.CPU()
        )
        assert data["2.float8_e4m3fn"].type == TensorType(
            DType.float8_e4m3fn, [2], device=DeviceRef.CPU()
        )
        assert data["2.float8_e5m2"].type == TensorType(
            DType.float8_e5m2, [2], device=DeviceRef.CPU()
        )


def test_load_using_prefix(testdata_directory) -> None:  # noqa: ANN001
    weights = SafetensorWeights(
        [
            testdata_directory / "example_data_1.safetensors",
            testdata_directory / "example_data_2.safetensors",
        ]
    )
    with Graph("test_load_safetensors_by_prefix") as graph:
        a = graph.add_weight(weights[1].a.allocate())
        assert a.type == TensorType(DType.int32, [5, 2], device=DeviceRef.CPU())
        b = graph.add_weight(weights["1.b"].allocate())
        assert b.type == TensorType(
            DType.float64, [1, 2, 3], device=DeviceRef.CPU()
        )


def test_load_same_weight(testdata_directory) -> None:  # noqa: ANN001
    weights = SafetensorWeights(
        [
            testdata_directory / "example_data_1.safetensors",
            testdata_directory / "example_data_2.safetensors",
        ]
    )
    with Graph("test_load_safetensors_same_weight") as graph:
        graph.add_weight(weights[1].a.allocate())
        with pytest.raises(ValueError, match="already exists"):
            graph.add_weight(weights[1]["a"].allocate())


def test_load_allocate_as_bytes(testdata_directory) -> None:  # noqa: ANN001
    weights = SafetensorWeights(
        [testdata_directory / "example_data_1.safetensors"]
    )
    with Graph("test_load_safetensors1") as graph:
        data = {
            key: graph.add_weight(weight.allocate_as_bytes())
            for key, weight in weights.items()
        }
        assert len(data) == 7
        assert data["1.a"].type == TensorType(
            DType.uint8, [5, 8], device=DeviceRef.CPU()
        )  # originally int32
        assert data["1.b"].type == TensorType(
            DType.uint8, [1, 2, 24], device=DeviceRef.CPU()
        )  # originally float64
        assert data["1.c"].type == TensorType(
            DType.uint8, [4], device=DeviceRef.CPU()
        )  # originally float32
        assert data["1.fancy/name"].type == TensorType(
            DType.uint8, [24], device=DeviceRef.CPU()
        )  # originally int64
        assert data["1.bf16"].type == TensorType(
            DType.uint8, [4], device=DeviceRef.CPU()
        )  # originally bfloat16
        assert data["1.float8_e4m3fn"].type == TensorType(
            DType.uint8, [2], device=DeviceRef.CPU()
        )  # originally float8_e4m3fn
        assert data["1.float8_e5m2"].type == TensorType(
            DType.uint8, [2], device=DeviceRef.CPU()
        )  # originally float8_e5m2


def test_load_invalid_tensor(testdata_directory) -> None:  # noqa: ANN001
    weights = SafetensorWeights(
        [testdata_directory / "example_data_1.safetensors"]
    )
    with pytest.raises(
        KeyError,
        match="'0.a' is not a weight in the Safetensor checkpoint. Did you mean '1.a'?",
    ):
        print(weights._tensors_to_file_idx)
        _ = weights["0"].a.allocate()
