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
import sys

import numpy as np
import pytest

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from addition import add_tensors


@pytest.mark.parametrize(
    "input0, input1, expected",
    [
        (
            np.array([2.0], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            np.array([5.0]),
        ),
        (
            np.array([-2.0], dtype=np.float32),
            np.array([-3.0], dtype=np.float32),
            np.array([-5.0]),
        ),
        (
            np.array([0.0], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
            np.array([5.0]),
        ),
        (
            np.array([1.23456], dtype=np.float32),
            np.array([2.34567], dtype=np.float32),
            np.array([3.58023]),
        ),
    ],
)
def test_add_tensors(input0, input1, expected) -> None:
    result = add_tensors(input0, input1)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_add_tensors_type() -> None:
    input0 = np.array([1.0], dtype=np.float32)
    input1 = np.array([2.0], dtype=np.float32)
    result = add_tensors(input0, input1)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_add_tensors_shape() -> None:
    input0 = np.array([1.0], dtype=np.float32)
    input1 = np.array([2.0], dtype=np.float32)
    result = add_tensors(input0, input1)
    assert result.shape == (1,)
