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
import numpy as np
import pytest
from max.driver import Accelerator, Device
from max.nn.parallel import ParallelArrayOps
from numpy.testing import assert_equal


@pytest.mark.parametrize("accelerator", [Accelerator(), None])
@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
@pytest.mark.parametrize(
    "shape", [(1000, 1000, 3), (1000, 1000), (1000, 1000, 3, 3)]
)
@pytest.mark.parametrize("num_arrays", [1, 2, 3, 4])
def test_parallel_concat(
    accelerator: Device | None,
    axis: int,
    shape: tuple[int, ...],
    num_arrays: int,
) -> None:
    parallel_ops = ParallelArrayOps(accelerator=accelerator)

    arrays = [np.random.rand(*shape) for _ in range(num_arrays)]

    # Validate axis is within bounds
    if axis < -len(shape) or axis >= len(shape):
        with pytest.raises(IndexError):
            _ = parallel_ops.concatenate(arrays, axis=axis)
        return

    out_max = parallel_ops.concatenate(arrays, axis=axis)
    out_np = np.concatenate(arrays, axis=axis)

    assert_equal(out_max.to_numpy(), out_np)
