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


@pytest.fixture(scope="module")
def shared_accelerator() -> Accelerator:
    return Accelerator()


@pytest.mark.parametrize("use_accelerator", [True, False])
@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_parallel_concat_multi_array(
    shared_accelerator: Accelerator,
    use_accelerator: bool,
    axis: int,
) -> None:
    """Multi-array concat. Exercises the main concat path across both the
    pinned accelerator buffer branch and the CPU/non-pinned ``Buffer``
    branch, and across axis positions covering the empty-prefix
    (``axis=0``), middle (``axis=1``), and negative-normalized
    (``axis=-1``) variants of the validation/output-shape logic."""
    accelerator: Device | None = shared_accelerator if use_accelerator else None
    parallel_ops = ParallelArrayOps(accelerator=accelerator)

    arrays = [np.random.rand(32, 32, 3) for _ in range(3)]

    out_max = parallel_ops.concatenate(arrays, axis=axis)
    out_np = np.concatenate(arrays, axis=axis)

    assert_equal(out_max.to_numpy(), out_np)


@pytest.mark.parametrize("use_accelerator", [True, False])
def test_parallel_concat_single_array(
    shared_accelerator: Accelerator,
    use_accelerator: bool,
) -> None:
    """Single-array (n==1) concat. Exercises the n==1 fast path across
    both branches: ``Buffer.from_numpy`` on CPU and ``DevicePinnedBuffer``
    + ``np.copyto`` on an accelerator."""
    accelerator: Device | None = shared_accelerator if use_accelerator else None
    parallel_ops = ParallelArrayOps(accelerator=accelerator)

    arr = np.random.rand(32, 32)

    out_max = parallel_ops.concatenate([arr], axis=0)

    assert_equal(out_max.to_numpy(), arr)


@pytest.mark.parametrize("axis", [2, -3])
def test_parallel_concat_axis_out_of_bounds(axis: int) -> None:
    """Axis outside the array dimensions raises IndexError, in both the
    positive-too-large and negative-too-small directions (the latter
    catches bugs in the ``axis = len(shape) + axis`` normalization)."""
    parallel_ops = ParallelArrayOps(accelerator=None)

    arrays = [np.random.rand(32, 32) for _ in range(2)]

    with pytest.raises(IndexError):
        _ = parallel_ops.concatenate(arrays, axis=axis)


def test_parallel_concat_split_path(
    shared_accelerator: Accelerator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the per-array chunk-splitting branch. Pins ``max_workers``
    and picks a ``min_chunk_size_mb`` well below the per-chunk size so
    the heuristic robustly splits regardless of internal default tweaks,
    then asserts that splitting actually happened by counting copy calls."""
    parallel_ops = ParallelArrayOps(
        accelerator=shared_accelerator, max_workers=8
    )

    # 64*64*4*8 = 131072 bytes per array; with max_workers=8 and n=2,
    # potential_workers_per_array=4 and min_chunk_bytes ~= 1049 bytes,
    # giving num_chunks=4 and chunk_bytes ~= 32768 (>>min). Robust margin.
    arrays = [np.random.rand(64, 64, 4) for _ in range(2)]

    call_count = 0
    original_copy = ParallelArrayOps._copy_array_slice

    def counting_copy(*args, **kwargs) -> None:
        nonlocal call_count
        call_count += 1
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(
        ParallelArrayOps,
        "_copy_array_slice",
        staticmethod(counting_copy),
    )

    out_max = parallel_ops.concatenate(arrays, axis=0, min_chunk_size_mb=0.001)
    out_np = np.concatenate(arrays, axis=0)

    assert_equal(out_max.to_numpy(), out_np)
    # If splitting happened we get more copy invocations than arrays.
    assert call_count > len(arrays), (
        f"expected chunk-split path (>{len(arrays)} copy calls), "
        f"got {call_count}"
    )
