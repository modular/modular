# M1.2 (#124) compiler-defect reproducer: `ioctl(fd, FIONREAD, &n)`
# declared as a fixed-arity `@extern("ioctl")` binding never actually
# delivers `&n`'s address to the kernel correctly. Observed TWO different
# failure shapes for the exact same code, depending on what happened to
# be sitting in the stack slot the real ioctl() reads its variadic
# argument from:
#   - in this standalone file: the call returns -1 with errno 14
#     (EFAULT) — the kernel rejects whatever address it actually read;
#   - in a larger program (spike/abi/libc_calls_test.mojo, same
#     declaration, same FIONREAD constant, more prior stack activity):
#     the call returns 0 (claimed SUCCESS) and the output cell is simply
#     never written — it stays at its initial sentinel no matter how many
#     times the call is retried, despite a real byte genuinely queued the
#     whole time.
# Both outcomes are consistent with the kernel receiving garbage instead
# of the real pointer: sometimes the garbage fails an address-validity
# check (EFAULT), sometimes it happens to look like a harmless-but-wrong
# address the ioctl still "succeeds" against without ever touching `n`.
# Undefined, stack-contents-dependent behavior for identical Mojo source
# is itself part of the finding.
#
# Bisected to the ARGUMENT ITSELF, not its declared type: the same
# outcome class happens whether `arg` is declared as a typed
# `UnsafePointer[Int32, MutAnyOrigin]` or as a raw `UInt64` machine word
# carrying the exact same address.
#
# Leading theory (consistent with every observation here, though not
# independently proven from Mojo's own compiler internals): Apple's arm64
# ABI requires arguments PAST a variadic function's last FIXED parameter
# to be passed on the STACK, never in a register. `ioctl`'s real C
# prototype is `int ioctl(int fd, unsigned long request, ...)` — only 2
# fixed parameters — so a genuine variadic call site puts `arg` on the
# stack. A Mojo `@extern` declaration has no way to mark a parameter as
# "the variadic tail"; it lowers all 3 declared parameters via the
# ordinary (register) calling convention. The real ioctl() implementation
# then reads its variadic argument via va_arg from the stack slot a
# non-variadic caller never populated — reading whatever garbage is
# there, which explains both observed outcomes above.
#
# fcntl (also nominally variadic, exercised in
# spike/abi/libc_calls_test.mojo) does NOT show this for its F_SETFL
# case: its variadic argument is a plain scalar `int`, and this repo's
# own test proves that value DOES reach the real fcntl(). Whether that
# is because a scalar happens to survive the mismatch, or because
# Darwin's fcntl() is implemented differently under the hood, is not
# established here.
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64. Requires a live TCP
# loopback pair, so this reproducer sets one up itself (no oracle dylib
# needed — every symbol here is a raw libc/OS call).
#
# Run: mojo run m1-2-ioctl-pointer-arg-lost.mojo
# Expected (bug, this file): NON-DETERMINISTIC across runs on this exact
# unmodified file — sometimes "ioctl rc: -1 errno: 14 avail: -99" (EFAULT),
# sometimes "ioctl rc: 0 errno: 0 avail: -99" (claimed success, still
# unwritten). Either way `avail` never becomes 1. This run-to-run
# variance for byte-identical source is itself the strongest evidence for
# the garbage-stack-slot theory above. A SEPARATE, likely unrelated
# flakiness was also observed on some runs: a b2 runtime crash AFTER the
# ioctl line prints ("recursive_mutex lock failed" inside
# libKGENCompilerRTShared.dylib during process exit) — consistent with
# the b2 toolchain's already-documented intermittent lowering/runtime
# crashes elsewhere in this repo (e.g. spike/completion/run.sh's retry-
# on-"Stack dump" convention); the ioctl finding itself is fully visible
# in the printed line before that crash, so it is not conflated with it.

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

    # Queue exactly 1 real byte via the stdlib's own direct write binding.
    var server_fd = FileDescriptor(Int(server))
    server_fd.write("X")

    var avail = stack_allocation[1, Int32]()
    avail[0] = -99
    var FIONREAD = UInt64(0x4004667f)  # macOS: _IOR('f', 127, int)
    var rc = mjo_ioctl(client, FIONREAD, avail)
    var ep = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(mjo_errno_ptr()))
    print("ioctl rc:", rc, "errno:", ep[0], "avail:", avail[0], "(expected avail: 1)")
