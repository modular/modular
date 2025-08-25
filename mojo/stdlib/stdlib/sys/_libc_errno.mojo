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
from sys.ffi import c_int, external_call
from sys.info import CompilationTarget, platform_map


fn get_errno() -> ErrNo:
    """Gets the current value of the libc errno.

    This function retrieves the thread-local errno value set by the last
    failed system call. The implementation is platform-specific, using
    `__errno_location()` on Linux and `__error()` on macOS.

    Returns:
        The current errno value as an ErrNo struct.

    Constrained:
        Compilation error on unsupported platforms.
    """

    @parameter
    if CompilationTarget.is_linux():
        return ErrNo(
            external_call["__errno_location", UnsafePointer[c_int]]()[]
        )
    elif CompilationTarget.is_macos():
        return ErrNo(external_call["__error", UnsafePointer[c_int]]()[])
    else:
        return CompilationTarget.unsupported_target_error[
            ErrNo, operation="get_errno"
        ]()


fn set_errno(errno: ErrNo):
    """Sets the C library errno to a specific value.

    This function sets the thread-local errno value. It's typically used to
    clear errno before making a system call to detect errors reliably.

    Args:
        errno: The errno value to set.

    Constrained:
        Compilation error on unsupported platforms.
    """

    @parameter
    if CompilationTarget.is_linux():
        external_call[
            "__errno_location", UnsafePointer[c_int]
        ]()[] = errno.value
    elif CompilationTarget.is_macos():
        external_call["__error", UnsafePointer[c_int]]()[] = errno.value
    else:
        CompilationTarget.unsupported_target_error[operation="set_errno"]()


# Alias to shorten the error definitions below
alias pm = platform_map


@fieldwise_init
@register_passable("trivial")
struct ErrNo(Copyable, EqualityComparable, Movable, Stringable, Writable):
    """Represents a error number from libc.

    This struct acts as an enum providing a wrapper around C library error codes,
    with platform-specific values for error constants.

    Example:
        ```mojo
        import os
        from sys.ffi import get_errno, set_errno, ErrNo

        try:
            _ = os.path.realpath("non-existant-file")
        except:
            var err = get_errno()
            if err == ErrNo.ENOENT:
                # Handle missing path, clear errno, and continue
                set_errno(ErrNo.SUCCESS)
            else:
                # Else raise error
                raise Error(err)
        ```
    """

    var value: c_int
    """The numeric error code value."""

    # fmt: off
    alias SUCCESS        = Self(0)
    "Success"
    alias EPERM          = Self(1)
    """Operation not permitted"""
    alias ENOENT         = Self(2)
    """No such file or directory"""
    alias ESRCH          = Self(3)
    """No such process"""
    alias EINTR          = Self(4)
    """Interrupted system call"""
    alias EIO            = Self(5)
    """I/O error"""
    alias ENXIO          = Self(6)
    """No such device or address"""
    alias E2BIG          = Self(7)
    """Argument list too long"""
    alias ENOEXEC        = Self(8)
    """Exec format error"""
    alias EBADF          = Self(9)
    """Bad file number"""
    alias ECHILD         = Self(10)
    """No child processes"""
    alias EAGAIN         = Self(pm["EAGAIN",           linux=11, macos=35]())
    """Try again"""
    alias ENOMEM         = Self(12)
    """Out of memory"""
    alias EACCES         = Self(13)
    """Permission denied"""
    alias EFAULT         = Self(14)
    """Bad address"""
    alias ENOTBLK        = Self(15)
    """Block device required"""
    alias EBUSY          = Self(16)
    """Device or resource busy"""
    alias EEXIST         = Self(17)
    """File exists"""
    alias EXDEV          = Self(18)
    """Cross-device link"""
    alias ENODEV         = Self(19)
    """No such device"""
    alias ENOTDIR        = Self(20)
    """Not a directory"""
    alias EISDIR         = Self(21)
    """Is a directory"""
    alias EINVAL         = Self(22)
    """Invalid argument"""
    alias ENFILE         = Self(23)
    """File table overflow"""
    alias EMFILE         = Self(24)
    """Too many open files"""
    alias ENOTTY         = Self(25)
    """Not a typewriter"""
    alias ETXTBSY        = Self(26)
    """Text file busy"""
    alias EFBIG          = Self(27)
    """File too large"""
    alias ENOSPC         = Self(28)
    """No space left on device"""
    alias ESPIPE         = Self(29)
    """Illegal seek"""
    alias EROFS          = Self(30)
    """Read-only file system"""
    alias EMLINK         = Self(31)
    """Too many links"""
    alias EPIPE          = Self(32)
    """Broken pipe"""
    alias EDOM           = Self(33)
    """Math argument out of domain of func"""
    alias ERANGE         = Self(34)
    """Math result not representable"""
    alias EDEADLK        = Self(pm["EDEADLK",          linux=35, macos=11]())
    """Resource deadlock would occur"""
    alias ENAMETOOLONG   = Self(pm["ENAMETOOLONG",     linux=36, macos=63]())
    """File name too long"""
    alias ENOLCK         = Self(pm["ENOLCK",           linux=37, macos=77]())
    """No record locks available"""
    alias ENOSYS         = Self(pm["ENOSYS",           linux=38, macos=78]())
    """Function not implemented"""
    alias ENOTEMPTY      = Self(pm["ENOTEMPTY",        linux=39, macos=66]())
    """Directory not empty"""
    alias ELOOP          = Self(pm["ELOOP",            linux=40, macos=62]())
    """Too many symbolic links encountered"""
    alias EWOULDBLOCK    = Self.EAGAIN
    """Operation would block"""
    alias ENOMSG         = Self(pm["ENOMSG",           linux=42, macos=91]())
    """No message of desired type"""
    alias EIDRM          = Self(pm["EIDRM",            linux=43, macos=90]())
    """Identifier removed"""
    alias ECHRNG         = Self(pm["ECHRNG",           linux=44]())
    """Channel number out of range"""
    alias EL2NSYNC       = Self(pm["EL2NSYNC",         linux=45]())
    """Level 2 not synchronized"""
    alias EL3HLT         = Self(pm["EL3HLT",           linux=46]())
    """Level 3 halted"""
    alias EL3RST         = Self(pm["EL3RST",           linux=47]())
    """Level 3 reset"""
    alias ELNRNG         = Self(pm["ELNRNG",           linux=48]())
    """Link number out of range"""
    alias EUNATCH        = Self(pm["EUNATCH",          linux=49]())
    """Protocol driver not attached"""
    alias ENOCSI         = Self(pm["ENOCSI",           linux=50]())
    """No CSI structure available"""
    alias EL2HLT         = Self(pm["EL2HLT",           linux=51]())
    """Level 2 halted"""
    alias EBADE          = Self(pm["EBADE",            linux=52]())
    """Invalid exchange"""
    alias EBADR          = Self(pm["EBADR",            linux=53]())
    """Invalid request descriptor"""
    alias EXFULL         = Self(pm["EXFULL",           linux=54]())
    """Exchange full"""
    alias ENOANO         = Self(pm["ENOANO",           linux=55]())
    """No anode"""
    alias EBADRQC        = Self(pm["EBADRQC",          linux=56]())
    """Invalid request code"""
    alias EBADSLT        = Self(pm["EBADSLT",          linux=57]())
    """Invalid slot"""
    alias EDEADLOCK      = Self.EDEADLK
    """Alias for EDEADLK"""
    alias EBFONT         = Self(pm["EBFONT",           linux=59]())
    """Bad font file format"""
    alias ENOSTR         = Self(pm["ENOSTR",           linux=60, macos=99]())
    """Device not a stream"""
    alias ENODATA        = Self(pm["ENODATA",          linux=61, macos=96]())
    """No data available"""
    alias ETIME          = Self(pm["ETIME",            linux=62, macos=101]())
    """Timer expired"""
    alias ENOSR          = Self(pm["ENOSR",            linux=63, macos=98]())
    """Out of streams resources"""
    alias ENONET         = Self(pm["ENONET",           linux=64]())
    """Machine is not on the network"""
    alias ENOPKG         = Self(pm["ENOPKG",           linux=65]())
    """Package not installed"""
    alias EREMOTE        = Self(pm["EREMOTE",          linux=66, macos=71]())
    """Object is remote"""
    alias ENOLINK        = Self(pm["ENOLINK",          linux=67, macos=97]())
    """Link has been severed"""
    alias EADV           = Self(pm["EADV",             linux=68]())
    """Advertise error"""
    alias ESRMNT         = Self(pm["ESRMNT",           linux=69]())
    """Srmount error"""
    alias ECOMM          = Self(pm["ECOMM",            linux=70]())
    """Communication error on send"""
    alias EPROTO         = Self(pm["EPROTO",           linux=71, macos=100]())
    """Protocol error"""
    alias EMULTIHOP      = Self(pm["EMULTIHOP",        linux=72, macos=95]())
    """Multihop attempted"""
    alias EDOTDOT        = Self(pm["EDOTDOT",          linux=73]())
    """RFS specific error"""
    alias EBADMSG        = Self(pm["EBADMSG",          linux=74, macos=94]())
    """Not a data message"""
    alias EOVERFLOW      = Self(pm["EOVERFLOW",        linux=75, macos=84]())
    """Value too large for defined data type"""
    alias ENOTUNIQ       = Self(pm["ENOTUNIQ",         linux=76]())
    """Name not unique on network"""
    alias EBADFD         = Self(pm["EBADFD",           linux=77]())
    """File descriptor in bad state"""
    alias EREMCHG        = Self(pm["EREMCHG",          linux=78]())
    """Remote address changed"""
    alias ELIBACC        = Self(pm["ELIBACC",          linux=79]())
    """Can not access a needed shared library"""
    alias ELIBBAD        = Self(pm["ELIBBAD",          linux=80]())
    """Accessing a corrupted shared library"""
    alias ELIBSCN        = Self(pm["ELIBSCN",          linux=81]())
    """.lib section in a.out corrupted"""
    alias ELIBMAX        = Self(pm["ELIBMAX",          linux=82]())
    """Attempting to link in too many shared libraries"""
    alias ELIBEXEC       = Self(pm["ELIBEXEC",         linux=83]())
    """Cannot exec a shared library directly"""
    alias EILSEQ         = Self(pm["EILSEQ",           linux=84, macos=92]())
    """Illegal byte sequence"""
    alias ERESTART       = Self(pm["ERESTART",         linux=85]())
    """Interrupted system call should be restarted"""
    alias ESTRPIPE       = Self(pm["ESTRPIPE",         linux=86]())
    """Streams pipe error"""
    alias EUSERS         = Self(pm["EUSERS",           linux=87, macos=68]())
    """Too many users"""
    alias ENOTSOCK       = Self(pm["ENOTSOCK",         linux=88, macos=38]())
    """Socket operation on non-socket"""
    alias EDESTADDRREQ   = Self(pm["EDESTADDRREQ",     linux=89, macos=39]())
    """Destination address required"""
    alias EMSGSIZE       = Self(pm["EMSGSIZE",         linux=90, macos=40]())
    """Message too long"""
    alias EPROTOTYPE     = Self(pm["EPROTOTYPE",       linux=91, macos=41]())
    """Protocol wrong type for socket"""
    alias ENOPROTOOPT    = Self(pm["ENOPROTOOPT",      linux=92, macos=42]())
    """Protocol not available"""
    alias EPROTONOSUPPORT= Self(pm["EPROTONOSUPPORT",  linux=93, macos=43]())
    """Protocol not supported"""
    alias ESOCKTNOSUPPORT= Self(pm["ESOCKTNOSUPPORT",  linux=94, macos=44]())
    """Socket type not supported"""
    alias EOPNOTSUPP     = Self(pm["EOPNOTSUPP",       linux=95, macos=102]())
    """Operation not supported on transport endpoint"""
    alias EPFNOSUPPORT   = Self(pm["EPFNOSUPPORT",     linux=96, macos=46]())
    """Protocol family not supported"""
    alias EAFNOSUPPORT   = Self(pm["EAFNOSUPPORT",     linux=97, macos=47]())
    """Address family not supported by protocol"""
    alias EADDRINUSE     = Self(pm["EADDRINUSE",       linux=98, macos=48]())
    """Address already in use"""
    alias EADDRNOTAVAIL  = Self(pm["EADDRNOTAVAIL",    linux=99, macos=49]())
    """Cannot assign requested address"""
    alias ENETDOWN       = Self(pm["ENETDOWN",         linux=100, macos=50]())
    """Network is down"""
    alias ENETUNREACH    = Self(pm["ENETUNREACH",      linux=101, macos=51]())
    """Network is unreachable"""
    alias ENETRESET      = Self(pm["ENETRESET",        linux=102, macos=52]())
    """Network dropped connection because of reset"""
    alias ECONNABORTED   = Self(pm["ECONNABORTED",     linux=103, macos=53]())
    """Software caused connection abort"""
    alias ECONNRESET     = Self(pm["ECONNRESET",       linux=104, macos=54]())
    """Connection reset by peer"""
    alias ENOBUFS        = Self(pm["ENOBUFS",          linux=105, macos=55]())
    """No buffer space available"""
    alias EISCONN        = Self(pm["EISCONN",          linux=106, macos=56]())
    """Transport endpoint is already connected"""
    alias ENOTCONN       = Self(pm["ENOTCONN",         linux=107, macos=57]())
    """Transport endpoint is not connected"""
    alias ESHUTDOWN      = Self(pm["ESHUTDOWN",        linux=108, macos=58]())
    """Cannot send after transport endpoint shutdown"""
    alias ETOOMANYREFS   = Self(pm["ETOOMANYREFS",     linux=109, macos=59]())
    """Too many references: cannot splice"""
    alias ETIMEDOUT      = Self(pm["ETIMEDOUT",        linux=110, macos=60]())
    """Connection timed out"""
    alias ECONNREFUSED   = Self(pm["ECONNREFUSED",     linux=111, macos=61]())
    """Connection refused"""
    alias EHOSTDOWN      = Self(pm["EHOSTDOWN",        linux=112, macos=64]())
    """Host is down"""
    alias EHOSTUNREACH   = Self(pm["EHOSTUNREACH",     linux=113, macos=65]())
    """No route to host"""
    alias EALREADY       = Self(pm["EALREADY",         linux=114, macos=37]())
    """Operation already in progress"""
    alias EINPROGRESS    = Self(pm["EINPROGRESS",      linux=115, macos=36]())
    """Operation now in progress"""
    alias ESTALE         = Self(pm["ESTALE",           linux=116, macos=70]())
    """Stale NFS file handle"""
    alias EUCLEAN        = Self(pm["EUCLEAN",          linux=117]())
    """Structure needs cleaning"""
    alias ENOTNAM        = Self(pm["ENOTNAM",          linux=118]())
    """Not a XENIX named type file"""
    alias ENAVAIL        = Self(pm["ENAVAIL",          linux=119]())
    """No XENIX semaphores available"""
    alias EISNAM         = Self(pm["EISNAM",           linux=120]())
    """Is a named type file"""
    alias EREMOTEIO      = Self(pm["EREMOTEIO",        linux=121]())
    """Remote I/O error"""
    alias EDQUOT         = Self(pm["EDQUOT",           linux=122, macos=69]())
    """Quota exceeded"""
    alias ENOMEDIUM      = Self(pm["ENOMEDIUM",        linux=123]())
    """No medium found"""
    alias EMEDIUMTYPE    = Self(pm["EMEDIUMTYPE",      linux=124]())
    """Wrong medium type"""
    alias ECANCELED      = Self(pm["ECANCELED",        linux=125, macos=89]())
    """Operation Canceled"""
    alias ENOKEY         = Self(pm["ENOKEY",           linux=126]())
    """Required key not available"""
    alias EKEYEXPIRED    = Self(pm["EKEYEXPIRED",      linux=127]())
    """Key has expired"""
    alias EKEYREVOKED    = Self(pm["EKEYREVOKED",      linux=128]())
    """Key has been revoked"""
    alias EKEYREJECTED   = Self(pm["EKEYREJECTED",     linux=129]())
    """Key was rejected by service"""
    alias EOWNERDEAD     = Self(pm["EOWNERDEAD",       linux=130, macos=105]())
    """Owner died"""
    alias ENOTRECOVERABLE= Self(pm["ENOTRECOVERABLE",  linux=131, macos=104]())
    """State not recoverable"""
    alias ERFKILL        = Self(pm["ERFKILL",          linux=132]())
    """Operation not possible due to RF-kill"""
    alias EHWPOISON      = Self(pm["EHWPOISON",        linux=133]())
    """Memory page has hardware error"""


    # macOS-specific
    alias ENOTSUP        = Self(pm["ENOTSUP",          macos=45]())
    """Operation not supported"""
    alias EPROCLIM       = Self(pm["EPROCLIM",         macos=67]())
    """Too many processes"""
    alias EBADRPC        = Self(pm["EBADRPC",          macos=72]())
    """RPC struct is bad"""
    alias ERPCMISMATCH   = Self(pm["ERPCMISMATCH",     macos=73]())
    """RPC version wrong"""
    alias EPROGUNAVAIL   = Self(pm["EPROGUNAVAIL",     macos=74]())
    """RPC prog. not avail"""
    alias EPROGMISMATCH  = Self(pm["EPROGMISMATCH",    macos=75]())
    """Program version wrong"""
    alias EPROCUNAVAIL   = Self(pm["EPROCUNAVAIL",     macos=76]())
    """Bad procedure for program"""
    alias EFTYPE         = Self(pm["EFTYPE",           macos=79]())
    """Inappropriate file type or format"""
    alias EAUTH          = Self(pm["EAUTH",            macos=80]())
    """Authentication error"""
    alias ENEEDAUTH      = Self(pm["ENEEDAUTH",        macos=81]())
    """Need authenticator"""
    alias EPWROFF        = Self(pm["EPWROFF",          macos=82]())
    """Device power is off"""
    alias EDEVERR        = Self(pm["EDEVERR",          macos=83]())
    """Device error, e.g. paper out"""
    alias EBADEXEC       = Self(pm["EBADEXEC",         macos=85]())
    """Bad executable"""
    alias EBADARCH       = Self(pm["EBADARCH",         macos=86]())
    """Bad CPU type in executable"""
    alias ESHLIBVERS     = Self(pm["ESHLIBVERS",       macos=87]())
    """Shared library version mismatch"""
    alias EBADMACHO      = Self(pm["EBADMACHO",        macos=88]())
    """Malformed Macho file"""
    alias ENOATTR        = Self(pm["ENOATTR",          macos=93]())
    """Attribute not found"""
    alias ENOPOLICY      = Self(pm["ENOPOLICY",        macos=103]())
    """No such policy registered"""
    alias EQFULL         = Self(pm["EQFULL",           macos=106]())
    """Interface output queue is full"""
    # fmt: on

    fn __init__(out self, value: Int):
        """Constructs an ErrNo from an integer value.

        Args:
            value: The numeric error code.
        """
        debug_assert(
            0 <= value <= Int(c_int.MAX),
            "constructed ErrNo from an `Int` out of range of `c_int`",
        )
        self.value = value

    fn write_to(self, mut writer: Some[Writer]):
        """Writes the human-readable error description to a writer.

        Args:
            writer: The writer to write the error description to.
        """

        @parameter
        if CompilationTarget.is_macos():
            debug_assert(
                self != ErrNo.SUCCESS, "macos can't stringify ErrNo.SUCCESS"
            )
        var ptr = external_call["strerror", UnsafePointer[Byte]](self.value)
        var string = StringSlice(unsafe_from_utf8_ptr=ptr)
        string.write_to(writer)

    fn __str__(self) -> String:
        """Returns the human-readable error description as a string.

        Returns:
            A string containing the error description from `strerror`.
        """
        return String.write(self)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Checks if two `ErrNo` values are equal.

        Args:
            other: The `ErrNo` value to compare with.

        Returns:
            True if the error codes are equal, False otherwise.
        """
        return self.value == other.value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Checks if two `ErrNo` values are not equal.

        Args:
            other: The `ErrNo` value to compare with.

        Returns:
            True if the error codes are not equal, False otherwise.
        """
        return self.value != other.value
