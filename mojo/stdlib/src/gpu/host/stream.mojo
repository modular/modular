# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA stream operations."""

from os import abort

from memory.unsafe import DTypePointer, Pointer

from ._utils import _check_error, _StreamHandle


@always_inline
fn _get_current_stream() -> DTypePointer[DType.invalid]:
    return external_call[
        "KGEN_CompilerRT_LLCL_GetCurrentStream", DTypePointer[DType.invalid]
    ]()


# ===----------------------------------------------------------------------===#
# Stream
# ===----------------------------------------------------------------------===#


struct Stream(CollectionElement):
    var stream: _StreamHandle
    var owning: Bool
    var cuda_dll: UnsafePointer[CudaDLL]

    @staticmethod
    fn get_current_stream() -> Stream:
        return Stream(_StreamHandle(_get_current_stream()))

    fn __init__(
        inout self,
        stream: _StreamHandle,
        cuda_dll: UnsafePointer[CudaDLL] = UnsafePointer[CudaDLL](),
    ):
        self.stream = stream
        self.owning = False
        self.cuda_dll = cuda_dll

    fn __init__(inout self, ctx: Context, stream: _StreamHandle):
        self.__init__(stream, ctx.cuda_dll)

    fn __init__(
        inout self,
        flags: Int = 0,
        cuda_dll: UnsafePointer[CudaDLL] = UnsafePointer[CudaDLL](),
    ) raises:
        var stream = _StreamHandle()
        self.cuda_dll = cuda_dll

        var cuStreamCreate = self.cuda_dll[].cuStreamCreate if self.cuda_dll else cuStreamCreate.load()
        _check_error(cuStreamCreate(Pointer.address_of(stream), Int32(0)))

        self.stream = stream
        self.owning = True

    fn __init__(inout self, ctx: Context, flags: Int = 0) raises:
        self.__init__(flags, ctx.cuda_dll)

    fn __del__(owned self):
        try:
            var cuStreamDestroy = self.cuda_dll[].cuStreamDestroy if self.cuda_dll else cuStreamDestroy.load()
            if self.owning and self.stream:
                _check_error(cuStreamDestroy(self.stream))
        except e:
            abort(e.__str__())

    fn __copyinit__(inout self, existing: Self):
        self.stream = existing.stream
        self.owning = False
        self.cuda_dll = existing.cuda_dll

    fn __moveinit__(inout self, owned existing: Self):
        self.stream = existing.stream
        self.owning = existing.owning
        self.cuda_dll = existing.cuda_dll
        existing.stream = _StreamHandle()
        existing.owning = False
        existing.cuda_dll = UnsafePointer[CudaDLL]()

    fn synchronize(self) raises:
        """Wait until a CUDA stream's tasks are completed."""
        if self.stream:
            var cuStreamSynchronize = self.cuda_dll[].cuStreamSynchronize if self.cuda_dll else cuStreamSynchronize.load()
            _check_error(cuStreamSynchronize(self.stream))
