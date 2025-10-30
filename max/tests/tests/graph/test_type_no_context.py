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
"""Tests type factoriy and accessor errors for when an MLIR context has not
been initialized."""

import pytest
from hypothesis import given
from max.graph import BufferType, Dim, TensorType, _ChainType, _OpaqueType


@given(dim=...)
def test_dim_to_mlir_no_context(dim: Dim) -> None:
    with pytest.raises(RuntimeError):
        print(dim.to_mlir())


@given(tensor_type=...)
def test_tensor_type_to_mlir_no_context(tensor_type: TensorType) -> None:
    with pytest.raises(RuntimeError):
        tensor_type.to_mlir()


def test_opaque_type_to_mlir_no_context() -> None:
    with pytest.raises(RuntimeError):
        _OpaqueType("something").to_mlir()


@given(buffer_type=...)
def test_buffer_mlir_roundtrip_no_context(buffer_type: BufferType) -> None:
    with pytest.raises(RuntimeError):
        buffer_type.to_mlir()


def test_chain_type__no_mlir_context() -> None:
    with pytest.raises(RuntimeError):
        _ChainType().to_mlir()
