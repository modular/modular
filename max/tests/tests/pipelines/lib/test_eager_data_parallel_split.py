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
"""Tests how data-parallel Llama splits a batch of real tensors."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.llama3_modulev3.llama3 import _split_batch
from max.tensor import Tensor


def _inputs(row_offsets: list[int], splits: list[int]) -> Any:
    """Returns the batch fields the split reads, as real tensors."""
    return SimpleNamespace(
        tokens=_tensor(np.arange(row_offsets[-1], dtype=np.int64)),
        input_row_offsets=_tensor(np.array(row_offsets, dtype=np.uint32)),
        host_input_row_offsets=_tensor(np.array(row_offsets, dtype=np.uint32)),
        data_parallel_splits=_tensor(np.array(splits, dtype=np.int64)),
    )


def _tensor(array: npt.NDArray[np.integer[Any]]) -> Tensor:
    return Tensor(storage=Buffer.from_numpy(array))


def test_each_replica_gets_its_sequences() -> None:
    (first_tokens, first_offsets), (second_tokens, second_offsets) = (
        _split_batch(
            [DeviceRef.CPU(), DeviceRef.CPU()], _inputs([0, 2, 5], [0, 1, 2])
        )
    )

    assert first_tokens.real and first_offsets.real
    assert first_tokens.to_numpy().tolist() == [0, 1]
    assert first_offsets.to_numpy().tolist() == [0, 2]
    assert second_tokens.to_numpy().tolist() == [2, 3, 4]
    assert second_offsets.to_numpy().tolist() == [0, 3]
    assert first_tokens.dtype == DType.int64
    assert first_offsets.dtype == DType.uint32


@pytest.mark.parametrize(
    ("splits", "empty", "full"), [([0, 2, 2], 1, 0), ([0, 0, 2], 0, 1)]
)
def test_a_replica_with_no_sequences_gets_no_rows(
    splits: list[int], empty: int, full: int
) -> None:
    replicas = _split_batch(
        [DeviceRef.CPU(), DeviceRef.CPU()], _inputs([0, 2, 5], splits)
    )

    tokens, offsets = replicas[empty]
    assert tokens.to_numpy().tolist() == []
    assert offsets.to_numpy().tolist() == [0]
    tokens, offsets = replicas[full]
    assert tokens.to_numpy().tolist() == [0, 1, 2, 3, 4]
    assert offsets.to_numpy().tolist() == [0, 2, 5]
