# Reproducer for: a fixed-arity `@extern("ioctl")` declaration of the
# variadic `ioctl(fd, FIONREAD, &n)` never delivers the pointer argument
# to the kernel. The out-cell is never written; the observed failure
# shape depends on what garbage happens to sit in the stack slot the
# real ioctl() reads its variadic argument from:
#   - rc -1 with errno 14 (EFAULT): the kernel rejects the address it
#     actually read;
#   - rc 0 (claimed success) with the cell untouched: the garbage
#     happened to look like an address the ioctl "succeeds" against.
# Within one binary the outcome is stable (10/10 identical runs in my
# re-testing); it varies across program contexts, not run to run.
#
# Cause (confirmed, not a theory): Apple's arm64 ABI passes arguments
# past a variadic function's last FIXED parameter on the stack, never in
# a register. `ioctl`'s real C prototype is
# `int ioctl(int fd, unsigned long request, ...)`, two fixed parameters,
# so a genuine variadic call site puts the third argument on the stack.
# An `@extern` declaration has no way to mark a parameter as the
# variadic tail, so it lowers all 3 declared parameters via the fixed
# (register) convention, and ioctl()'s own va_arg reads a stack slot
# nothing populated. Confirmed three ways:
#   1. A C program calling ioctl through a fixed 3-arg function pointer
#      fails identically (rc -1 / EFAULT / cell untouched) while the
#      proper variadic call succeeds, on the same host.
#   2. Mojo 1.0.0 fixed this exact class for external_call (commit
#      8e783b63, "Support variadic C functions in external_call"):
#      external_call["ioctl", Int32, num_fixed_args=2](fd, FIONREAD, p)
#      delivers the pointer correctly (verified: the cell becomes 1).
#   3. This file's fixed-arity @extern declaration still silently
#      miscompiles on 1.0.0 stable (5/5 runs).
#
# What remains open in the linked issue is the @extern-side gap: no way
# to declare a variadic tail, and no diagnostic when a fixed-arity
# declaration targets a known-variadic symbol.
#
# `fcntl`'s scalar F_SETFL argument happening to survive the same
# mis-declaration on this target is consistent with the mechanism: the
# garbage only bites when the callee reads a slot the caller never
# populated in the right place.
#
# Verified on: Mojo 1.0.0b2 (2cf4d08a) and 1.0.0 (ed45d567), macOS
# arm64. Requires a live TCP loopback pair, so this reproducer sets one
# up itself (every symbol here is a raw libc/OS call).
#
# Run: mojo run repro.mojo
# Expected (bug): "avail: -99" — the cell is never written, whatever
# rc/errno pair this context produces. Expected (fixed): "avail: 1".
# Separately, some runs crash AFTER the ioctl line prints
# ("recursive_mutex lock failed" during process exit); that is a known
# unrelated b2 runtime flake and does not affect the printed finding.

from std.io import FileDescriptor
from std.memory import stack_allocation

comptime ByteBuf = UnsafePointer[Byte, MutAnyOrigin]
comptime I32Slot = UnsafePointer[Int32, MutAnyOrigin]

@extern("socket")
def mjo_socket(domain: Int32, sock_type: Int32, protocol: Int32) abi("C") -> Int32: ...
@extern("bind")
def mjo_bind(fd: Int32, addr: ByteBuf, len: UInt32) abi("C") -> Int32: ...
@extern("listen")
def mjo_listen(fd: Int32, backlog: Int32) abi("C") -> Int32: ...
@extern("connect")
def mjo_connect(fd: Int32, addr: ByteBuf, len: UInt32) abi("C") -> Int32: ...
@extern("accept")
def mjo_accept(fd: Int32, addr: ByteBuf, len: ByteBuf) abi("C") -> Int32: ...
@extern("getsockname")
def mjo_getsockname(fd: Int32, addr: ByteBuf, len: ByteBuf) abi("C") -> Int32: ...
@extern("__error")
def mjo_errno_ptr() abi("C") -> UInt64: ...
@extern("usleep")
def mjo_usleep(usec: UInt32) abi("C") -> Int32: ...
@extern("ioctl")
def mjo_ioctl(fd: Int32, request: UInt64, arg: I32Slot) abi("C") -> Int32: ...


def main() raises:
    var listener = mjo_socket(2, 1, 0)  # AF_INET, SOCK_STREAM
    var addr_buf = stack_allocation[16, Byte]()
    for i in range(16):
        addr_buf[i] = 0
    addr_buf.bitcast[Int16]()[0] = 2  # AF_INET, host byte order low byte
    _ = mjo_bind(listener, addr_buf.bitcast[Byte](), 16)
    _ = mjo_listen(listener, 1)
    var len_buf = stack_allocation[1, UInt32]()
    len_buf[0] = 16
    _ = mjo_getsockname(listener, addr_buf.bitcast[Byte](), len_buf.bitcast[Byte]())
    var client = mjo_socket(2, 1, 0)
    _ = mjo_connect(client, addr_buf.bitcast[Byte](), 16)
    var dummy1 = stack_allocation[16, Byte]()
    var dummy2 = stack_allocation[1, UInt32]()
    var server = mjo_accept(listener, dummy1.bitcast[Byte](), dummy2.bitcast[Byte]())

    # Queue exactly 1 real byte via the stdlib's own direct write binding,
    # then give loopback delivery a moment so FIONREAD has something to
    # count (matters once the call is fixed; without the settle delay a
    # correctly delivered FIONREAD can legitimately report 0).
    var server_fd = FileDescriptor(Int(server))
    server_fd.write("X")
    _ = mjo_usleep(100000)

    var avail = stack_allocation[1, Int32]()
    avail[0] = -99
    var FIONREAD = UInt64(0x4004667f)  # macOS: _IOR('f', 127, int)
    var rc = mjo_ioctl(client, FIONREAD, avail)
    var ep = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(mjo_errno_ptr()))
    print("ioctl rc:", rc, "errno:", ep[0], "avail:", avail[0], "(expected avail: 1)")
