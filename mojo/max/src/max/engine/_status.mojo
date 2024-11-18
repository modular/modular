# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from memory import UnsafePointer
from sys.ffi import DLHandle
from max._utils import call_dylib_func, exchange, CString


@value
@register_passable("trivial")
struct CStatus:
    """Represents Status ptr from Engine."""

    var ptr: UnsafePointer[NoneType]

    alias IsErrorFnName = "M_isError"
    alias GetErrorFnName = "M_getError"
    alias FreeStatusFnName = "M_freeStatus"

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn is_error(self, lib: DLHandle) -> Bool:
        """
        Check if status is error.

        Returns:
            True if error.
        """
        return call_dylib_func[Bool](lib, Self.IsErrorFnName, self)

    fn get_error(self, lib: DLHandle) -> String:
        """
        Get Error String from Engine library.
        """
        var error = call_dylib_func[CString](lib, Self.GetErrorFnName, self)
        return error.__str__()

    fn free(self, lib: DLHandle):
        """
        Free the status ptr.
        """
        call_dylib_func(lib, Self.FreeStatusFnName, self)


struct Status:
    var ptr: CStatus
    var lib: DLHandle

    alias NewStatusFnName = "M_newStatus"

    @implicit
    fn __init__(out self, lib: DLHandle):
        self.ptr = call_dylib_func[CStatus](lib, self.NewStatusFnName)
        self.lib = lib

    fn __moveinit__(out self, owned existing: Self):
        self.ptr = exchange[CStatus](existing.ptr, UnsafePointer[NoneType]())
        self.lib = existing.lib

    fn __bool__(self) -> Bool:
        """
        Check if status is error.

        Returns:
            True if error.
        """
        return self.ptr.is_error(self.lib)

    fn __str__(self) -> String:
        """
        Get Error String.

        Returns:
            Error string if there is an error. Else empty.
        """
        if self:
            return self.ptr.get_error(self.lib)
        return ""

    fn borrow_ptr(self) -> CStatus:
        """
        Borrow the underlying C ptr.
        """
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
