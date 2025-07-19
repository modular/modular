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
# RUN: %mojo %s

from testing import assert_equal, assert_false, assert_raises, assert_true

from sys.ffi.c._libc import set_errno, get_errno
from sys.ffi.c.constants import *

alias error_message = (
    (SUCCESS, "Success"),
    (EPERM, "Operation not permitted"),
    (ENOENT, "No such file or directory"),
    (ESRCH, "No such process"),
    (EINTR, "Interrupted system call"),
    (EIO, "Input/output error"),
    (ENXIO, "No such device or address"),
    (E2BIG, "Argument list too long"),
    (ENOEXEC, "Exec format error"),
    (EBADF, "Bad file descriptor"),
    (ECHILD, "No child processes"),
    (EAGAIN, "Resource temporarily unavailable"),
    (EWOULDBLOCK, "Resource temporarily unavailable"),
    (ENOMEM, "Cannot allocate memory"),
    (EACCES, "Permission denied"),
    (EFAULT, "Bad address"),
    (ENOTBLK, "Block device required"),
    (EBUSY, "Device or resource busy"),
    (EEXIST, "File exists"),
    (EXDEV, "Invalid cross-device link"),
    (ENODEV, "No such device"),
    (ENOTDIR, "Not a directory"),
    (EISDIR, "Is a directory"),
    (EINVAL, "Invalid argument"),
    (ENFILE, "Too many open files in system"),
    (EMFILE, "Too many open files"),
    (ENOTTY, "Inappropriate ioctl for device"),
    (ETXTBSY, "Text file busy"),
    (EFBIG, "File too large"),
    (ENOSPC, "No space left on device"),
    (ESPIPE, "Illegal seek"),
    (EROFS, "Read-only file system"),
    (EMLINK, "Too many links"),
    (EPIPE, "Broken pipe"),
    (EDOM, "Numerical argument out of domain"),
    (ERANGE, "Numerical result out of range"),
    (EDEADLK, "Resource deadlock avoided"),
    (ENAMETOOLONG, "File name too long"),
    (ENOLCK, "No locks available"),
    (ENOSYS, "Function not implemented"),
    (ENOTEMPTY, "Directory not empty"),
    (ELOOP, "Too many levels of symbolic links"),
    (ENOMSG, "No message of desired type"),
    (EIDRM, "Identifier removed"),
    (ECHRNG, "Channel number out of range"),
    (EL2NSYNC, "Level 2 not synchronized"),
    (EL3HLT, "Level 3 halted"),
    (EL3RST, "Level 3 reset"),
    (ELNRNG, "Link number out of range"),
    (EUNATCH, "Protocol driver not attached"),
    (ENOCSI, "No CSI structure available"),
    (EL2HLT, "Level 2 halted"),
    (EBADE, "Invalid exchange"),
    (EBADR, "Invalid request descriptor"),
    (EXFULL, "Exchange full"),
    (ENOANO, "No anode"),
    (EBADRQC, "Invalid request code"),
    (EBADSLT, "Invalid slot"),
    (EBFONT, "Bad font file format"),
    (ENOSTR, "Device not a stream"),
    (ENODATA, "No data available"),
    (ETIME, "Timer expired"),
    (ENOSR, "Out of streams resources"),
    (ENONET, "Machine is not on the network"),
    (ENOPKG, "Package not installed"),
    (EREMOTE, "Object is remote"),
    (ENOLINK, "Link has been severed"),
    (EADV, "Advertise error"),
    (ESRMNT, "Srmount error"),
    (ECOMM, "Communication error on send"),
    (EPROTO, "Protocol error"),
    (EMULTIHOP, "Multihop attempted"),
    (EDOTDOT, "RFS specific error"),
    (EBADMSG, "Bad message"),
    (EOVERFLOW, "Value too large for defined data type"),
    (ENOTUNIQ, "Name not unique on network"),
    (EBADFD, "File descriptor in bad state"),
    (EREMCHG, "Remote address changed"),
    (ELIBACC, "Can not access a needed shared library"),
    (ELIBBAD, "Accessing a corrupted shared library"),
    (ELIBSCN, ".lib section in a.out corrupted"),
    (ELIBMAX, "Attempting to link in too many shared libraries"),
    (ELIBEXEC, "Cannot exec a shared library directly"),
    (EILSEQ, "Invalid or incomplete multibyte or wide character"),
    (ERESTART, "Interrupted system call should be restarted"),
    (ESTRPIPE, "Streams pipe error"),
    (EUSERS, "Too many users"),
    (ENOTSOCK, "Socket operation on non-socket"),
    (EDESTADDRREQ, "Destination address required"),
    (EMSGSIZE, "Message too long"),
    (EPROTOTYPE, "Protocol wrong type for socket"),
    (ENOPROTOOPT, "Protocol not available"),
    (EPROTONOSUPPORT, "Protocol not supported"),
    (ESOCKTNOSUPPORT, "Socket type not supported"),
    (EOPNOTSUPP, "Operation not supported"),
    (EPFNOSUPPORT, "Protocol family not supported"),
    (EAFNOSUPPORT, "Address family not supported by protocol"),
    (EADDRINUSE, "Address already in use"),
    (EADDRNOTAVAIL, "Cannot assign requested address"),
    (ENETDOWN, "Network is down"),
    (ENETUNREACH, "Network is unreachable"),
    (ENETRESET, "Network dropped connection on reset"),
    (ECONNABORTED, "Software caused connection abort"),
    (ECONNRESET, "Connection reset by peer"),
    (ENOBUFS, "No buffer space available"),
    (EISCONN, "Transport endpoint is already connected"),
    (ENOTCONN, "Transport endpoint is not connected"),
    (ESHUTDOWN, "Cannot send after transport endpoint shutdown"),
    (ETOOMANYREFS, "Too many references: cannot splice"),
    (ETIMEDOUT, "Connection timed out"),
    (ECONNREFUSED, "Connection refused"),
    (EHOSTDOWN, "Host is down"),
    (EHOSTUNREACH, "No route to host"),
    (EALREADY, "Operation already in progress"),
    (EINPROGRESS, "Operation now in progress"),
    (ESTALE, "Stale file handle"),
    (EUCLEAN, "Structure needs cleaning"),
    (ENOTNAM, "Not a XENIX named type file"),
    (ENAVAIL, "No XENIX semaphores available"),
    (EISNAM, "Is a named type file"),
    (EREMOTEIO, "Remote I/O error"),
    (EDQUOT, "Disk quota exceeded"),
    (ENOMEDIUM, "No medium found"),
    (EMEDIUMTYPE, "Wrong medium type"),
    (ECANCELED, "Operation canceled"),
    (ENOKEY, "Required key not available"),
    (EKEYEXPIRED, "Key has expired"),
    (EKEYREVOKED, "Key has been revoked"),
    (EKEYREJECTED, "Key was rejected by service"),
    (EOWNERDEAD, "Owner died"),
    (ENOTRECOVERABLE, "State not recoverable"),
    (ERFKILL, "Operation not possible due to RF-kill"),
    (EHWPOISON, "Memory page has hardware error"),
)


def test_errno():
    @parameter
    for i in range(len(error_message)):
        errno_msg = error_message[i]
        errno = errno_msg[0]
        set_errno(i)
        assert_equal(get_errno(), i)
    set_errno(0)


def main():
    test_errno()
