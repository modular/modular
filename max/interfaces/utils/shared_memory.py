# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Shared memory utilities for zero-copy NumPy array transfer."""

from __future__ import annotations

import logging
import os
import uuid
import weakref
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def can_allocate(size: int) -> bool:
    """Check if we can allocate the given size in shared memory.

    Args:
        size: Size in bytes to check

    Returns:
        True if allocation is likely to succeed
    """
    try:
        stat = os.statvfs(path="/dev/shm")
        available = stat.f_bsize * stat.f_bavail
    except OSError:
        # If we can't check capacity, assume we can allocate.
        return True

    watermark = float(os.getenv("MODULAR_MAX_SHM_WATERMARK", "0.8"))
    return size < available * watermark


class SharedMemoryArray:
    """Wrapper for numpy array stored in shared memory.

    This class is used as a placeholder in pixel_values during serialization.
    It will be encoded as a dict with __shm__ flag and decoded back to a numpy
    array.
    """

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype


def ndarray_to_shared_memory(arr: npt.NDArray[Any]) -> SharedMemoryArray | None:
    """Convert a NumPy array to shared memory and return a reference descriptor.

    Includes capacity checking to prevent exhausting /dev/shm.

    Args:
        arr: The NumPy array to store in shared memory

    Returns:
        SharedMemoryArray if successful, None if shared memory is full or creation fails
    """
    # Check shared memory capacity.
    if not can_allocate(arr.nbytes) or arr.nbytes == 0:
        return None

    try:
        # Generate a unique name for this shared memory segment.
        name = f"maximg-{uuid.uuid4().hex}"

        # Create shared memory segment with exact size needed.
        shm = shared_memory.SharedMemory(
            create=True, size=arr.nbytes, name=name
        )

        # Copy array data into shared memory.
        shm_arr: npt.NDArray[Any] = np.ndarray(
            arr.shape, arr.dtype, buffer=shm.buf
        )

        # Handle 0-dimensional arrays (scalars) differently
        if arr.ndim == 0:
            shm_arr[()] = arr
        else:
            shm_arr[:] = arr

        # Close our handle but don't unlink - let the consumer handle cleanup.
        shm.close()

        return SharedMemoryArray(
            name=name, shape=arr.shape, dtype=arr.dtype.str
        )

    except (OSError, FileExistsError) as e:
        logger.warning(f"Shared memory creation failed: {e}")
        return None


def open_shm_array(meta: dict[str, Any]) -> npt.NDArray[Any]:
    """Open a shared memory array.

    Args:
        meta: Dictionary with 'name', 'shape', and 'dtype' keys

    Returns:
        NumPy array either as a view of the shared memory
    """
    shm = shared_memory.SharedMemory(name=meta["name"])

    # Create numpy array view into shared memory
    arr: npt.NDArray[Any] = np.ndarray(
        shape=meta["shape"], dtype=np.dtype(meta["dtype"]), buffer=shm.buf
    )

    # Mode: register cleanup and mark for deletion when last reference closes.
    weakref.finalize(arr, shm.close)
    shm.unlink()

    # NOTE: we could reduce shared memory pressure by returning a copy here.
    return arr
