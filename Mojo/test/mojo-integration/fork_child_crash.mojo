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

# A forked child that crashes under `mojo run` must die by its own signal, as
# it does in a `mojo build` binary, so the parent's `waitpid` sees that signal.
# The sanitizers install their own fatal-signal handlers, which the re-raised
# signal reaches in place of the default action.
# UNSUPPORTED: asan, ubsan
# RUN: %mojo %s | FileCheck %s

from std.ffi import external_call
from std.memory import Pointer
from std.time import perf_counter_ns, sleep

comptime WNOHANG: Int32 = 1
comptime SIGKILL: Int32 = 9


def child_signal(sig: Int32, deadline_ns: Int) -> String:
    var pid = external_call["fork", Int32]()
    if pid == 0:
        _ = external_call["kill", Int32](external_call["getpid", Int32](), sig)
        external_call["_exit", NoneType](Int32(0))
    var status = Int32(0)
    # A child that hangs in the crash handler must fail the test, not stall it.
    while (
        external_call["waitpid", Int32](pid, Pointer(to=status), WNOHANG) == 0
    ):
        if perf_counter_ns() > deadline_ns:
            _ = external_call["kill", Int32](pid, SIGKILL)
            _ = external_call["waitpid", Int32](
                pid, Pointer(to=status), Int32(0)
            )
            return "timeout"
        sleep(0.01)
    return String(status & 0x7F)


def main():
    var deadline_ns = perf_counter_ns() + 60_000_000_000
    # CHECK: SIGABRT: 6
    print("SIGABRT:", child_signal(6, deadline_ns))
    # CHECK: SIGILL: 4
    print("SIGILL:", child_signal(4, deadline_ns))
    # CHECK: SIGFPE: 8
    print("SIGFPE:", child_signal(8, deadline_ns))
    # CHECK: SIGSEGV: 11
    print("SIGSEGV:", child_signal(11, deadline_ns))
