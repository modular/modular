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
"""``Array`` — a numpy-style view over a HAL ``Buffer``.

An ``Array`` adds dtype, shape, strides, and a byte offset on top of the
bytes-only HAL :class:`~max.sys._hal.Buffer`.
"""

from __future__ import annotations

import math
import numbers
import os
from collections.abc import Iterator, Sequence
from itertools import product
from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt
from max.dtype import DType

from .buffer import Buffer
from .context import Context
from .queue import Queue


def _row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Returns the row-major (C-contiguous) strides for ``shape``, in elements.

    The empty shape (a rank-0 scalar) yields an empty tuple.
    """
    strides = [1] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def _packed_byte_len(dtype: DType, num_elements: int) -> int:
    """Returns the packed byte size of ``num_elements`` values of ``dtype``.

    Sub-byte dtypes (``int4``, ``float4``) are packed densely, so the size is
    ``ceil(num_elements * bits / 8)``. For byte-multiple dtypes this is just
    ``num_elements * dtype.size_in_bytes``.
    """
    return (num_elements * dtype.size_in_bits + 7) // 8


def _contiguous_suffix(
    shape: Sequence[int], strides: Sequence[int]
) -> tuple[int, int]:
    """Splits axes into a strided outer prefix and a contiguous inner run.

    Walks axes from innermost outward; an axis joins the contiguous run when
    its stride equals the running row-major stride. Size-1 axes always join
    (they contribute a single position and never move the pointer).

    Returns ``(split, run_len)`` where axes ``[split:]`` form one contiguous
    block of ``run_len`` elements.
    """
    expected = 1
    run_len = 1
    split = len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        if shape[axis] == 1:
            split = axis
            continue
        if strides[axis] != expected:
            break
        run_len *= shape[axis]
        expected *= shape[axis]
        split = axis
    return split, run_len


class Array:
    """A dtype + shape + strides view over a HAL ``Buffer``.

    Not constructed directly; use :meth:`empty`, :meth:`full`, or
    :meth:`from_numpy`, or slice an existing ``Array``.
    """

    _buffer: Buffer
    _context: Context
    _dtype: DType
    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _byte_offset: int
    _pinned: bool
    _queue: Queue | None

    __slots__ = (
        "_buffer",
        "_byte_offset",
        "_context",
        "_dtype",
        "_pinned",
        "_queue",
        "_shape",
        "_strides",
    )

    def __init__(self) -> None:
        raise TypeError(
            "Array is not directly constructible; use Array.empty(...), "
            "Array.full(...), Array.from_numpy(...), or slice an Array."
        )

    @classmethod
    def _make(
        cls,
        *,
        buffer: Buffer,
        context: Context,
        dtype: DType,
        shape: Sequence[int],
        strides: Sequence[int],
        byte_offset: int,
        pinned: bool,
    ) -> Array:
        obj = cls.__new__(cls)
        obj._buffer = buffer
        obj._context = context
        obj._dtype = dtype
        obj._shape = tuple(shape)
        obj._strides = tuple(strides)
        obj._byte_offset = int(byte_offset)
        obj._pinned = pinned
        obj._queue = None
        return obj

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(
        cls,
        context: Context,
        dtype: DType,
        shape: Sequence[int] = (),
        *,
        pinned: bool = False,
    ) -> Array:
        """Allocates an uninitialized contiguous array.

        With no ``shape`` the result is a rank-0 (scalar) array holding one
        element.
        """
        shape = tuple(int(d) for d in shape)
        byte_size = _packed_byte_len(dtype, math.prod(shape))
        buffer = (
            context.alloc_host_pinned(byte_size)
            if pinned
            else context.alloc_sync(byte_size)
        )
        return cls._make(
            buffer=buffer,
            context=context,
            dtype=dtype,
            shape=shape,
            strides=_row_major_strides(shape),
            byte_offset=0,
            pinned=pinned,
        )

    @classmethod
    def full(
        cls,
        context: Context,
        dtype: DType,
        shape: Sequence[int] = (),
        value: float | int = 0,
        *,
        pinned: bool = False,
    ) -> Array:
        """Allocates a contiguous array and fills it with ``value``.

        With no ``shape`` the result is a rank-0 (scalar) array; ``value``
        defaults to zero.
        """
        arr = cls.empty(context, dtype, shape, pinned=pinned)
        arr.fill(value)
        return arr

    @classmethod
    def from_numpy(
        cls,
        context: Context,
        np_array: npt.NDArray[Any],
    ) -> Array:
        """Allocates a device array from ``np_array`` (blocking H2D copy).

        Args:
            context: The context whose device the array is allocated on.
            np_array: The source array.

        Returns:
            A new contiguous ``Array`` holding the data.
        """
        host = np.ascontiguousarray(np_array)
        dtype = DType.from_numpy(host.dtype)
        arr = cls.empty(context, dtype, host.shape)
        if host.nbytes:
            context.copy_to_device_sync(arr._buffer.view(), host.ctypes.data)
        return arr

    @classmethod
    def from_list(
        cls,
        context: Context,
        values: npt.ArrayLike,
        dtype: DType | None = None,
    ) -> Array:
        """Allocates a device array from Python values, copying them (H2D).

        Accepts a scalar, a (possibly nested) list/tuple, or any array-like;
        the shape follows the nesting. A convenience over :meth:`from_numpy`
        for callers that don't want to build a numpy array themselves.

        Args:
            context: The context whose device the array is allocated on.
            values: The values to fill the array with.
            dtype: The element type. Defaults to whatever numpy infers from
                ``values``; pass a ``DType`` to force it. The dtype must have a
                numpy equivalent.

        Returns:
            A new contiguous ``Array`` holding a copy of ``values``.
        """
        np_dtype = None if dtype is None else dtype.to_numpy()
        return cls.from_numpy(context, np.asarray(values, dtype=np_dtype))

    @classmethod
    def from_file(
        cls,
        context: Context,
        path: PathLike[str] | str,
        dtype: DType,
        shape: Sequence[int],
        *,
        offset: int = 0,
    ) -> Array:
        """Allocates a device array from a binary file's raw bytes.

        The file is memory-mapped read-only, its packed bytes are copied to
        the device (blocking H2D), and the mapping is dropped. Because the
        bytes are copied raw, any dtype works — including ones with no numpy
        equivalent (``bfloat16``, ``float4``).

        Args:
            context: The context whose device the array is allocated on.
            path: The binary file holding the packed element bytes.
            dtype: The element type of the stored data.
            shape: The shape of the stored data.
            offset: Byte offset into the file where the data starts.

        Returns:
            A new contiguous ``Array`` holding the file's data.

        Raises:
            ValueError: If the file is too short for ``offset`` plus the
                packed byte size of ``dtype`` and ``shape``.
        """
        shape = tuple(int(d) for d in shape)
        nbytes = _packed_byte_len(dtype, math.prod(shape))
        available = os.path.getsize(path) - offset
        if available < nbytes:
            raise ValueError(
                f"file {os.fspath(path)!r} holds {available} bytes past "
                f"offset {offset}, but dtype {dtype} with shape {shape} "
                f"needs {nbytes}"
            )
        arr = cls.empty(context, dtype, shape)
        if nbytes:
            host = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,)
            )
            context.copy_to_device_sync(arr._buffer.view(), host.ctypes.data)
        return arr

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Strides in elements (row-major for a contiguous array)."""
        return self._strides

    @property
    def rank(self) -> int:
        return len(self._shape)

    @property
    def num_elements(self) -> int:
        return math.prod(self._shape)

    @property
    def element_size(self) -> int:
        """Per-element size in bytes, rounded up for sub-byte dtypes.

        For allocation or copy math use :attr:`byte_size`, which packs sub-byte
        dtypes densely; this rounds a sub-byte element up to one byte.
        """
        return self._dtype.size_in_bytes

    @property
    def byte_size(self) -> int:
        """Packed in-memory size of the array's elements in bytes.

        Sub-byte dtypes are packed densely (``ceil(num_elements * bits / 8)``);
        for byte-multiple dtypes this equals ``num_elements * element_size``.
        """
        return _packed_byte_len(self._dtype, self.num_elements)

    @property
    def byte_offset(self) -> int:
        """Byte offset of this array's first element into its ``Buffer``."""
        return self._byte_offset

    @property
    def is_contiguous(self) -> bool:
        return self._strides == _row_major_strides(self._shape)

    @property
    def buffer(self) -> Buffer:
        return self._buffer

    @property
    def context(self) -> Context:
        return self._context

    @property
    def pinned(self) -> bool:
        return self._pinned

    @property
    def data_ptr(self) -> int:
        """Backend address of this array's first element.

        Passes through the HAL's notion of the allocation's address (a device
        virtual address on CUDA/HIP, a host-visible address on Metal) plus the
        array's byte offset; never a fabricated value.
        """
        base = self._context.memory_get_address(self._buffer)
        return base + self._byte_offset

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def fill(self, value: float | int) -> None:
        """Fills every element with ``value`` (blocking).

        Encodes ``value`` to its raw bytes, then writes them across the array.
        If those bytes are all identical (any zero fill included), a single-byte
        memset is used; otherwise the full dtype-width pattern is written. A
        strided array is filled one contiguous run at a time.
        """
        raw = self._encode_fill(value)
        queue = self._get_queue()
        if len(set(raw)) == 1:
            byte = raw[0]
            for src_byte, run_bytes in self._runs():
                if run_bytes:
                    view = self._buffer.view(
                        byte_offset=src_byte, byte_size=run_bytes
                    )
                    queue.set_memory(view, byte)
        else:
            packed, value_size = int.from_bytes(raw, "little"), len(raw)
            if value_size not in (1, 2, 4, 8):
                # Plugins only fill 1/2/4/8-byte patterns; Metal silently
                # no-ops other widths, so reject rather than corrupt.
                raise ValueError(
                    f"cannot fill dtype {self._dtype}: pattern width "
                    f"{value_size} is not one of 1, 2, 4, or 8 bytes"
                )
            for src_byte, run_bytes in self._runs():
                if run_bytes:
                    view = self._buffer.view(
                        byte_offset=src_byte, byte_size=run_bytes
                    )
                    queue.fill(view, packed, value_size)
        queue.synchronize()

    @staticmethod
    def copy(src: Array, dst: Array) -> None:
        """Copies ``src`` into ``dst`` on the same device (blocking).

        Shapes and element sizes must match. A strided ``src`` is materialized
        contiguous first; a strided ``dst`` is written run by run. Runs directly
        on the context with no queue.
        """
        if src.shape != dst.shape:
            raise ValueError(f"copy shape mismatch: {src.shape} != {dst.shape}")
        if src.dtype.size_in_bits != dst.dtype.size_in_bits:
            raise ValueError(
                "copy element-size mismatch: "
                f"{src.dtype.size_in_bits} != {dst.dtype.size_in_bits} bits"
            )
        if src._context is not dst._context:
            raise ValueError(
                "copy requires both arrays on the same context; "
                "cross-context copies are not supported"
            )
        source = src if src.is_contiguous else src.contiguous()
        context = dst._context
        src_byte = source._byte_offset
        for dst_byte, run_bytes in dst._runs():
            if run_bytes:
                context.copy_intra_device_sync(
                    dst._buffer.view(byte_offset=dst_byte, byte_size=run_bytes),
                    source._buffer.view(
                        byte_offset=src_byte, byte_size=run_bytes
                    ),
                )
                src_byte += run_bytes

    def as_numpy(self) -> npt.NDArray[Any]:
        """Returns the array's contents as a numpy array (blocking D2H copy).

        A strided array is materialized contiguous first.

        Returns:
            A numpy array holding the data.

        Raises:
            ValueError: If the dtype has no numpy equivalent (e.g. ``bfloat16``).
        """
        source = self if self.is_contiguous else self.contiguous()
        out = np.empty(self._shape, dtype=self._dtype.to_numpy())
        if out.nbytes:
            source._context.copy_from_device_sync(
                out.ctypes.data,
                source._buffer.view(
                    byte_offset=source._byte_offset, byte_size=out.nbytes
                ),
            )
        return out

    def view(
        self, dtype: DType | None = None, shape: Sequence[int] | None = None
    ) -> Array:
        """Reinterprets the bytes as ``dtype`` and/or ``shape``, sharing the
        same ``Buffer``.

        Requires a contiguous array. ``dtype`` defaults to this array's own
        dtype, so a pure reshape needs only ``shape``. ``shape`` may contain
        one ``-1`` entry, whose size is inferred from the total byte size.
        With no ``shape`` and a different ``dtype``, the last axis is
        rescaled by the element-size ratio (numpy ``ndarray.view`` semantics).

        Args:
            dtype: The element type to reinterpret as. Defaults to this
                array's own dtype.
            shape: The new shape, with at most one ``-1`` entry to infer.
                Defaults to a dtype-rescaled version of the current shape.

        Returns:
            A new ``Array`` sharing this array's ``Buffer``.
        """
        if not self.is_contiguous:
            raise ValueError(
                "view requires a contiguous array; call contiguous() first"
            )
        dtype = self._dtype if dtype is None else dtype
        old_bits = self._dtype.size_in_bits
        new_bits = dtype.size_in_bits
        total_bits = self.num_elements * old_bits

        if shape is not None:
            dims = [int(d) for d in shape]
            if sum(1 for d in dims if d < 0) > 1 or any(d < -1 for d in dims):
                raise ValueError(
                    f"view shape {tuple(dims)} may contain at most one -1"
                )
            if -1 in dims:
                known = math.prod(d for d in dims if d != -1)
                if known == 0 or total_bits % (known * new_bits) != 0:
                    raise ValueError(
                        f"cannot infer view shape {tuple(dims)} of dtype "
                        f"{dtype} from {total_bits} bits"
                    )
                dims[dims.index(-1)] = total_bits // (known * new_bits)
            new_shape = tuple(dims)
            if math.prod(new_shape) * new_bits != total_bits:
                raise ValueError(
                    f"view shape {new_shape} of dtype {dtype} is "
                    f"{math.prod(new_shape) * new_bits} bits, but the array "
                    f"is {total_bits} bits"
                )
        elif old_bits == new_bits:
            new_shape = self._shape
        elif self._shape:
            last_bits = self._shape[-1] * old_bits
            if last_bits % new_bits != 0:
                raise ValueError(
                    f"cannot view last axis of {self._shape[-1]} {old_bits}-"
                    f"bit elements as {new_bits}-bit elements"
                )
            new_shape = self._shape[:-1] + (last_bits // new_bits,)
        else:
            raise ValueError(
                f"cannot view a rank-0 {old_bits}-bit array as a "
                f"{new_bits}-bit dtype without an explicit shape"
            )

        if (self._byte_offset * 8) % new_bits != 0:
            raise ValueError(
                f"byte offset {self._byte_offset} is not aligned to the "
                f"{new_bits}-bit view dtype"
            )
        return Array._make(
            buffer=self._buffer,
            context=self._context,
            dtype=dtype,
            shape=new_shape,
            strides=_row_major_strides(new_shape),
            byte_offset=self._byte_offset,
            pinned=self._pinned,
        )

    def reshape(self, shape: Sequence[int]) -> Array:
        """Returns this array reshaped to ``shape``, sharing the same
        ``Buffer``. A shorthand for :meth:`view` without a dtype change.

        Requires a contiguous array. ``shape`` may contain one ``-1`` entry,
        whose size is inferred from the current shape.
        """
        return self.view(shape=shape)

    def contiguous(self) -> Array:
        """Returns a contiguous copy, or ``self`` if already contiguous.

        The copy is built up one contiguous run at a time: the largest
        contiguous inner block is copied for each outer position, using the
        HAL same-device copy.
        """
        if self.is_contiguous:
            return self
        dst = Array.empty(
            self._context, self._dtype, self._shape, pinned=self._pinned
        )
        context = self._context
        dst_byte = 0
        for src_byte, run_bytes in self._runs():
            if run_bytes:
                context.copy_intra_device_sync(
                    dst._buffer.view(byte_offset=dst_byte, byte_size=run_bytes),
                    self._buffer.view(
                        byte_offset=src_byte, byte_size=run_bytes
                    ),
                )
                dst_byte += run_bytes
        return dst

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, index: object) -> Array:
        """numpy basic indexing, returning a view sharing the same ``Buffer``.

        Supports integers (drop a dim, fold into the offset), slices (including
        negative steps), tuples thereof, ``Ellipsis``, and implicit trailing
        full slices. A full integer index returns a rank-0 ``Array`` (not a
        scalar). ``newaxis`` / ``None`` is not supported.
        """
        selectors = self._normalize_index(index)
        bits = self._dtype.size_in_bits
        new_shape: list[int] = []
        new_strides: list[int] = []
        offset_bits = self._byte_offset * 8
        for axis, selector in enumerate(selectors):
            dim = self._shape[axis]
            stride = self._strides[axis]
            if isinstance(selector, slice):
                start, stop, step = selector.indices(dim)
                length = len(range(start, stop, step))
                new_shape.append(length)
                new_strides.append(step * stride)
                offset_bits += start * stride * bits
            else:  # integer
                canonical = selector + dim if selector < 0 else selector
                if canonical < 0 or canonical >= dim:
                    raise IndexError(
                        f"index {selector} is out of bounds for axis {axis} "
                        f"with size {dim}"
                    )
                offset_bits += canonical * stride * bits
        if offset_bits % 8 != 0:
            # Only reachable for sub-byte dtypes: a byte-addressed view cannot
            # begin mid-byte. Byte-multiple dtypes are always byte-aligned.
            raise ValueError(
                f"cannot index dtype {self._dtype} at a bit-unaligned offset: "
                "a sub-byte slice must begin on a byte boundary"
            )
        return Array._make(
            buffer=self._buffer,
            context=self._context,
            dtype=self._dtype,
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            byte_offset=offset_bits // 8,
            pinned=self._pinned,
        )

    def __setitem__(self, index: object, value: Array | npt.ArrayLike) -> None:
        """Assigns into the region selected by ``index`` (numpy basic indexing).

        A scalar ``value`` fills the selected region; an ``Array`` or array-like
        of the exact selected shape is copied into it. Broadcasting beyond a
        scalar and fancy indexing are not supported. The write goes through to
        the base allocation, so assigning into a slice mutates this array.
        """
        target = self[index]
        if isinstance(value, (int, float)):
            target.fill(value)
            return
        if isinstance(value, Array):
            src = value
        else:
            src = Array.from_numpy(
                target._context,
                np.asarray(value, dtype=target._dtype.to_numpy()),
            )
        if tuple(src.shape) != target.shape:
            raise ValueError(
                f"cannot assign shape {tuple(src.shape)} into shape "
                f"{target.shape}; broadcasting beyond a scalar is not supported"
            )
        Array.copy(src, target)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_queue(self) -> Queue:
        if self._queue is None:
            self._queue = self._context.create_queue()
        return self._queue

    def _encode_fill(self, value: float | int) -> bytes:
        # Filling with +0 is all-zero bytes in every dtype, so we can shortcut
        # and skip numpy. That matters because some dtypes (bfloat16, float4)
        # have no numpy equivalent, so the numpy path below would fail for them.
        # -0.0 does not qualify: it sets the sign bit rather than being all
        # zeros, so it takes the numpy path to keep the sign.
        is_positive_zero = value == 0 and not (
            isinstance(value, float) and math.copysign(1.0, value) < 0.0
        )
        if is_positive_zero:
            return b"\x00" * self.element_size
        return np.array(value, dtype=self._dtype.to_numpy()).tobytes()

    def _runs(self) -> Iterator[tuple[int, int]]:
        """Yields ``(buffer_byte_offset, run_bytes)`` for each contiguous run.

        Runs cover the array's elements in row-major order; offsets are
        absolute into ``self._buffer``.
        """
        split, run_len = _contiguous_suffix(self._shape, self._strides)
        run_bytes = _packed_byte_len(self._dtype, run_len)
        bits = self._dtype.size_in_bits
        outer_shape = self._shape[:split]
        outer_strides = self._strides[:split]
        base_bits = self._byte_offset * 8
        for outer in product(*(range(d) for d in outer_shape)):
            elem = sum(
                i * s for i, s in zip(outer, outer_strides, strict=False)
            )
            offset_bits = base_bits + elem * bits
            if offset_bits % 8 != 0:
                # Only reachable for a strided sub-byte array whose runs do not
                # start on byte boundaries; a byte-addressed copy cannot express
                # a mid-byte start.
                raise ValueError(
                    f"cannot address dtype {self._dtype} at a bit-unaligned "
                    "run offset: strided sub-byte access must be byte-aligned"
                )
            yield offset_bits // 8, run_bytes

    def _normalize_index(self, index: object) -> list[int | slice]:
        """Turns a subscript into one selector per axis.

        Whatever was passed to ``arr[...]`` becomes a list with exactly one
        ``int`` or ``slice`` for every axis, so :meth:`__getitem__` can handle
        each axis on its own. This fills in an ``Ellipsis`` and any omitted
        trailing axes with full slices.

        Args:
            index: The object passed to ``arr[index]``.

        Returns:
            A list of ``int`` and ``slice`` selectors, one per axis.
        """
        rank = len(self._shape)
        items: tuple[object, ...] = (
            index if isinstance(index, tuple) else (index,)
        )

        if any(item is Ellipsis for item in items):
            if sum(1 for item in items if item is Ellipsis) > 1:
                raise IndexError(
                    "an index can only have a single ellipsis ('...')"
                )
            consumed = sum(
                1 for item in items if item is not Ellipsis and item is not None
            )
            fill = rank - consumed
            if fill < 0:
                raise IndexError("too many indices for array")
            expanded: list[object] = []
            for item in items:
                if item is Ellipsis:
                    expanded.extend([slice(None)] * fill)
                else:
                    expanded.append(item)
            items = tuple(expanded)

        selectors: list[int | slice] = []
        for item in items:
            if isinstance(item, bool):
                raise TypeError("boolean indexing is not supported")
            elif isinstance(item, slice):
                selectors.append(item)
            elif isinstance(item, numbers.Integral):
                selectors.append(int(item))
            elif item is None:
                raise TypeError("newaxis / None indexing is not supported")
            else:
                raise TypeError(
                    f"unsupported index type: {type(item).__name__}"
                )

        if len(selectors) > rank:
            raise IndexError(
                f"too many indices for array: got {len(selectors)}, "
                f"array is rank {rank}"
            )
        while len(selectors) < rank:
            selectors.append(slice(None))
        return selectors

    def __repr__(self) -> str:
        return (
            f"Array(shape={self._shape}, dtype={self._dtype}, "
            f"strides={self._strides}, "
            f"contiguous={self.is_contiguous})"
        )

    __str__ = __repr__
