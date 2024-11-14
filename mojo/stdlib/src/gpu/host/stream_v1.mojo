# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA stream operations."""

from os import abort

from gpu.host.context_v1 import Context
from gpu.host.cuda_instance_v1 import CudaDLL
from gpu.host._utils_v1 import _check_error, _StreamHandle

# ===----------------------------------------------------------------------===#
# Stream
# ===----------------------------------------------------------------------===#


@value
struct Stream(CollectionElement):
    var stream: _StreamHandle
    var owning: Bool
    var cuda_dll: CudaDLL

    fn __init__(out self, stream: _StreamHandle, cuda_dll: CudaDLL):
        self.stream = stream
        self.owning = False
        self.cuda_dll = cuda_dll

    fn __init__(out self, ctx: Context, stream: _StreamHandle):
        self.__init__(stream, ctx.cuda_dll)

    fn __init__(
        inout self,
        cuda_dll: CudaDLL,
        flags: Int = 0,
    ) raises:
        self.stream = _StreamHandle()
        self.cuda_dll = cuda_dll
        self.owning = True

        _check_error(
            self.cuda_dll.cuStreamCreate(
                UnsafePointer.address_of(self.stream), Int32(flags)
            )
        )

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn __init__(out self, ctx: Context, flags: Int = 0) raises:
        self.__init__(ctx.cuda_dll, flags)

    fn __del__(owned self):
        try:
            if self.owning and self.stream:
                _check_error(self.cuda_dll.cuStreamDestroy(self.stream))
        except e:
            abort(e.__str__())

    fn __copyinit__(out self, existing: Self):
        self.stream = existing.stream
        self.owning = False
        self.cuda_dll = existing.cuda_dll

    fn synchronize(self) raises:
        """Wait until a CUDA stream's tasks are completed."""
        if self.stream:
            _check_error(self.cuda_dll.cuStreamSynchronize(self.stream))
