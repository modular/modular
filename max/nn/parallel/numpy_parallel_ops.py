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
"""Utility classes for parallelized numpy operations."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import numpy.typing as npt


class ParallelArrayOps:
    """Parallelized numpy array operations for performance-critical data processing.

    Uses ThreadPoolExecutor to parallelize bulk copy operations that release the GIL,
    enabling true multi-threaded execution. Particularly effective for concatenating
    large arrays where memory bandwidth can be saturated across multiple cores.

    Thread pool cleanup is handled automatically via __del__ and weakref.finalize,
    providing defense-in-depth for resource cleanup.

    Example:
        >>> ops = ParallelArrayOps(max_workers=20)
        >>> result = ops.concatenate([arr1, arr2, arr3], axis=0)
    """

    def __init__(self, max_workers: int = 24) -> None:
        """Initialize parallel array operations with a thread pool.

        Args:
            max_workers: Maximum number of worker threads. Default is 24, which works
                well for typical server CPUs. Consider setting to match your expected
                number of arrays (e.g., 20 for up to 20 concurrent copies).
        """
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers

        self._shutdown = False
        self._shutdown_lock = threading.Lock()

        # Register weakref finalizer as a safety net for cleanup
        self._finalizer = weakref.finalize(
            self, self._finalize_shutdown, self._pool, self._shutdown_lock
        )

    @staticmethod
    def _finalize_shutdown(
        pool: ThreadPoolExecutor, shutdown_lock: threading.Lock
    ) -> None:
        """Static cleanup method called by weakref.finalize.

        Args:
            pool: The ThreadPoolExecutor to shutdown.
            shutdown_lock: Lock for thread-safe shutdown.
        """
        try:
            with shutdown_lock:
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            # Suppress errors during finalization
            pass

    def __del__(self) -> None:
        """Cleanup method called when the instance is being destroyed."""
        try:
            self.shutdown(wait=False)
        except Exception:
            # Suppress errors during cleanup
            pass

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool and release resources.

        Args:
            wait: If True, wait for pending tasks to complete. If False, cancel
                pending tasks immediately.
        """
        with self._shutdown_lock:
            if self._shutdown:
                return

            self._shutdown = True
            try:
                self._pool.shutdown(wait=wait, cancel_futures=not wait)
            except Exception:
                pass

    def concatenate(
        self,
        arrays: Sequence[npt.NDArray[Any]],
        axis: int = 0,
        default_copy: bool = True,
    ) -> npt.NDArray[Any]:
        """Concatenate arrays in parallel along the specified axis.

        Equivalent to np.concatenate but parallelized using thread pool. Most effective
        with large arrays (>1MB each) and multiple arrays (>= max_workers).

        Note: By default, this matches np.concatenate's behavior of always returning a copy,
        even when arrays contains a single array. For performance optimization, you can set
        default_copy=False to avoid the arbitrary unnecessary copy and return the single
        array directly without copying.

        Args:
            arrays: List of numpy arrays to concatenate. Must have compatible shapes
                (same shape except along concatenation axis) and identical dtypes.
            axis: Axis along which to concatenate. Negative values count from the end.
                Default is 0.
            default_copy: If True (default), return a copy when arrays contains a single
                array, matching np.concatenate's behavior. If False, return the array
                directly without copying to avoid unnecessary memory allocation and
                improve performance.

        Returns:
            Concatenated array with the same dtype as the input arrays.

        Raises:
            ValueError: If arrays is empty, shapes are incompatible, or dtypes differ.
            RuntimeError: If the thread pool has been shut down.
        """
        n = len(arrays)
        if n == 0:
            raise ValueError("Cannot concatenate empty list of arrays.")

        if n == 1:
            # This copy is likely not needed, but it mocks the exact behaviour of numpy.concatenate.
            if default_copy:
                return arrays[0].copy()
            else:
                return arrays[0]

        # Validate shapes and compute output shape
        first_shape = arrays[0].shape
        first_dtype = arrays[0].dtype

        # Normalize negative axis and validate
        if axis < 0:
            axis = len(first_shape) + axis

        if not (0 <= axis < len(first_shape)):
            raise IndexError(
                f"axis {axis} is out of bounds for array of dimension {len(first_shape)}"
            )

        # Pre-compute expected shape slices for efficient validation
        # All arrays must match: shape[:axis] and shape[axis+1:]
        expected_prefix = first_shape[:axis]
        expected_suffix = first_shape[axis + 1 :]

        # Validate dtype and compute total size along concatenation axis in a single pass
        concat_dim_size = 0
        offsets = [0] * (n + 1)

        for i, arr in enumerate(arrays):
            # Check dtype (fast equality check, fails fast)
            if arr.dtype != first_dtype:
                raise ValueError(
                    f"All arrays must have the same dtype. "
                    f"arrays[0]={first_dtype}, arrays[{i}]={arr.dtype}"
                )

            # Validate shape compatibility using tuple slicing (much faster than dimension-by-dimension)
            # Fast path: if shapes are identical, both comparisons pass immediately
            arr_shape = arr.shape
            if (
                arr_shape[:axis] != expected_prefix
                or arr_shape[axis + 1 :] != expected_suffix
            ):
                # Find specific dimension for detailed error message
                for dim in range(len(first_shape)):
                    if dim != axis and arr_shape[dim] != first_shape[dim]:
                        raise ValueError(
                            f"All arrays must have same shape except along concat axis. "
                            f"Dimension {dim}: arrays[0]={first_shape[dim]}, arrays[{i}]={arr_shape[dim]}"
                        )

            concat_dim_size += arr.shape[axis]
            offsets[i + 1] = concat_dim_size

        # Check if all arrays are contiguous
        # Non-contiguous arrays significantly slow down performance.
        arrays = [
            np.ascontiguousarray(arr) if not arr.flags["C_CONTIGUOUS"] else arr
            for arr in arrays
        ]

        # Create output shape
        out_shape = list(first_shape)
        out_shape[axis] = concat_dim_size
        out = np.empty(out_shape, dtype=first_dtype, order="C")

        # Divide work among threads
        workers = min(self._max_workers, n)
        step = (n + workers - 1) // workers

        work_items = [(i, min(i + step, n)) for i in range(0, n, step)]

        futures = [
            self._pool.submit(
                self._copy_block_concatenate,
                out,
                arrays,
                offsets,
                start_idx,
                end_idx,
                axis,
            )
            for start_idx, end_idx in work_items
        ]

        # Wait for completion and on first failure cancel remaining tasks
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                raise

        return out

    @staticmethod
    def _copy_block_concatenate(
        out: npt.NDArray[Any],
        arrays: Sequence[npt.NDArray[Any]],
        offsets: Sequence[int],
        start_idx: int,
        end_idx: int,
        axis: int,
    ) -> None:
        """Worker function that copies a contiguous block of arrays into output.

        Runs in a worker thread. Uses np.copyto which releases the GIL, enabling
        true parallel execution across multiple CPU cores.

        Args:
            out: Pre-allocated output array.
            arrays: List of source arrays.
            offsets: Cumulative offsets along concatenation axis.
            start_idx: Starting index in arrays list (inclusive).
            end_idx: Ending index in arrays list (exclusive).
            axis: Axis along which arrays are concatenated.
        """
        # Pre-compute slice list to avoid allocating on every iteration
        out_slice = [slice(None)] * len(out.shape)

        for i in range(start_idx, end_idx):
            # Only update the slice for the concatenation axis
            out_slice[axis] = slice(offsets[i], offsets[i + 1])

            # Bulk copy - this releases the GIL
            if axis == 0:
                np.copyto(out[offsets[i] : offsets[i + 1]], arrays[i])
            elif axis == len(out.shape) - 1:
                np.copyto(out[..., offsets[i] : offsets[i + 1]], arrays[i])
            else:
                np.copyto(out[tuple(out_slice)], arrays[i])
