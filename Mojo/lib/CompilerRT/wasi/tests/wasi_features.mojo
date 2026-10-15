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
# wasm64 WASI stdlib checks. Prints PASS/FAIL and exits nonzero on failure.
# Paths are relative to the runner's preopen.

from std import os
from std.ffi import external_call
from std.simd import SIMD
from std.io import FileHandle
from std.math import isclose
from std.pathlib import cwd
from std.random import random_float64, seed
from std.sys import argv, exit, size_of
from std.sys._libc_errno import ErrNo, get_errno, set_errno
from std.time import monotonic, perf_counter_ns, sleep
from std.time.time import _process_cputime_nanoseconds, _realtime_nanoseconds


@fieldwise_init
struct _c_pollfd(Copyable):
    """Matches wasi-libc's struct pollfd (8 bytes on wasm64)."""

    var fd: Int32
    var events: Int16
    var revents: Int16


def check(cond: Bool, name: StaticString, mut failures: Int):
    if cond:
        print("PASS:", name)
    else:
        print("FAIL:", name)
        failures += 1


def main() raises:
    var failures = 0

    var dir = "."
    var path = "known.txt"
    try:
        os.remove(path)
    except:
        pass

    try:
        os.rmdir("subdir")
    except:
        pass
    os.mkdir("subdir")
    check(os.path.isdir("subdir"), "mkdir creates a directory", failures)

    var content = "hello from wasm64 file io\n"
    var f = FileHandle(path, "w")
    f.write(content)
    f.close()
    var g = FileHandle(path, "r")
    var back = g.read()
    g.close()
    check(back == content, "file create/write/read roundtrip", failures)

    var st = os.stat(path)
    check(st.st_size == content.byte_length(), "stat st_size", failures)
    # S_IFMT 0o170000, S_IFREG 0o100000.
    check(
        st.st_mode & 0o170000 == 0o100000, "stat st_mode regular file", failures
    )
    var lst = os.lstat(path)
    check(lst.st_size == st.st_size, "lstat agrees with stat", failures)

    var now_ns = _realtime_nanoseconds()
    var mtime_ns = st.st_mtimespec.as_nanoseconds()
    var day_ns = 86_400 * 1_000_000_000
    check(
        mtime_ns > now_ns - day_ns and mtime_ns < now_ns + day_ns,
        "stat mtime ns plausible",
        failures,
    )

    var entries = os.listdir(dir)
    check(
        len(entries) == 2 and "known.txt" in entries and "subdir" in entries,
        "listdir known contents",
        failures,
    )

    set_errno(ErrNo.SUCCESS)
    var raised = False
    try:
        var h = FileHandle("does-not-exist.txt", "r")
        h.close()
    except:
        raised = True
    check(raised, "open nonexistent raises", failures)
    check(
        Int(get_errno().value) == 44,
        "errno is WASI ENOENT (44)",
        failures,
    )

    var t0 = monotonic()
    var t1 = t0
    var spins = 0
    while t1 == t0 and spins < 10_000_000:
        t1 = monotonic()
        spins += 1
    check(t1 > t0, "monotonic clock advances", failures)
    var p0 = perf_counter_ns()
    var p1 = perf_counter_ns()
    check(p1 >= p0, "perf_counter_ns works", failures)
    var c0 = _process_cputime_nanoseconds()
    var c1 = _process_cputime_nanoseconds()
    check(c0 >= 0 and c1 >= c0, "process cputime advances", failures)

    # sleep goes through wasi-libc nanosleep -> WASI poll_oneoff.
    var s0 = monotonic()
    sleep(0.01)
    var s1 = monotonic()
    check(s1 - s0 >= 10_000_000, "sleep advances monotonic time", failures)

    # poll(2) -> wasi-libc poll -> WASI poll_oneoff: a regular file reports
    # readable (wasi-libc POLLIN 0x1); an invalid fd reports POLLNVAL (0x4000).
    comptime assert size_of[_c_pollfd]() == 8, "pollfd layout must be 8 bytes"
    var pf_file = FileHandle(path, "r")
    var pf = _c_pollfd(Int32(pf_file.handle), Int16(0x001), Int16(0))
    var nready = external_call["poll", Int32](Pointer(to=pf), Int(1), Int32(0))
    check(
        nready == 1 and (pf.revents & Int16(0x001)) != 0,
        "poll reports regular file readable",
        failures,
    )
    var bad = _c_pollfd(Int32(99), Int16(0x001), Int16(0))
    var nbad = external_call["poll", Int32](Pointer(to=bad), Int(1), Int32(0))
    check(
        nbad == 1 and (bad.revents & Int16(0x4000)) != 0,
        "poll reports POLLNVAL for invalid fd",
        failures,
    )
    pf_file.close()

    # SIMD lane arithmetic: the wasm targets default +simd128 (LLVM's wasm
    # defaults leave it off, which scalarizes every vector op). Per-lane
    # runtime values defeat constant folding; lanes are checked against a
    # scalar reference within a small tolerance.
    var simd_seed = Float32((monotonic() % 97) + 1)
    var va = SIMD[DType.float32, 8](
        simd_seed,
        simd_seed * 1.5,
        simd_seed * 2.0,
        simd_seed * 2.5,
        simd_seed * 3.0,
        simd_seed * 3.5,
        simd_seed * 4.0,
        simd_seed * 4.5,
    )
    var vb = SIMD[DType.float32, 8](
        simd_seed + 1.0,
        simd_seed + 2.0,
        simd_seed + 3.0,
        simd_seed + 4.0,
        simd_seed + 5.0,
        simd_seed + 6.0,
        simd_seed + 7.0,
        simd_seed + 8.0,
    )
    var vc = va + vb
    var want_vec = SIMD[DType.float32, 8](0)
    for i in range(8):
        # va[i] = simd_seed * (2 + i) / 2; vb[i] = simd_seed + 1 + i.
        want_vec[i] = (
            simd_seed * (Float32(i) + 2.0) / 2.0 + simd_seed + Float32(i) + 1.0
        )
    # atol-only match of the 0.001 tolerance; isclose fails NaN lanes by
    # default (equal_nan=False).
    check(
        all(isclose(vc, want_vec, atol=1e-3, rtol=0)),
        "SIMD float32 vector add matches scalar reference",
        failures,
    )

    # Fused mul+add chains: with the +relaxed-simd default, LLVM contracts
    # these into relaxed ops (e.g. f32x4.relaxed_madd) that the engine may
    # execute as a single FMA. Results must still match the scalar reference
    # within tolerance (relaxed ops permit engine-dependent rounding).
    var vsum = (va * vb) + va
    var want_fused = SIMD[DType.float32, 8](0)
    for i in range(8):
        # va[i] = s*(2+i)/2; vb[i] = s+1+i; want = va[i]*vb[i] + va[i].
        want_fused[i] = (
            simd_seed
            * (Float32(i) + 2.0)
            / 2.0
            * (simd_seed + Float32(i) + 1.0)
            + simd_seed * (Float32(i) + 2.0) / 2.0
        )
    check(
        all(isclose(vsum, want_fused, atol=1e-3, rtol=0)),
        "SIMD float32 fused mul-add matches scalar reference",
        failures,
    )

    # Every random draw resolves the global RNG state through the shim's
    # by-name globals table (hashed lookup); re-seeding must reproduce the
    # same sequence.
    seed(42)
    var r1 = random_float64()
    var r2 = random_float64()
    seed(42)
    var r3 = random_float64()
    var r4 = random_float64()
    check(
        r1 == r3 and r2 == r4 and r1 != r2,
        "seeded random sequence reproducible",
        failures,
    )

    check(
        os.getenv("WASI_FEATURES_UNSET_VAR", "fallback") == "fallback",
        "getenv returns default for unset var",
        failures,
    )
    check(
        os.setenv("WASI_FEATURES_PROBE", "probe-value"),
        "setenv succeeds",
        failures,
    )
    check(
        os.getenv("WASI_FEATURES_PROBE") == "probe-value",
        "getenv reads back setenv value",
        failures,
    )
    check(os.unsetenv("WASI_FEATURES_PROBE"), "unsetenv succeeds", failures)
    check(
        os.getenv("WASI_FEATURES_PROBE", "gone") == "gone",
        "getenv after unsetenv returns default",
        failures,
    )

    var args = argv()
    check(len(args) >= 1, "argv has at least argv[0]", failures)
    check(
        String(args[0]).endswith("wasi_features.wasm"),
        "argv[0] is the program name",
        failures,
    )

    var cwd_str = String(cwd())
    check(
        cwd_str.byte_length() >= 1 and cwd_str.startswith("/"),
        "cwd returns an absolute path",
        failures,
    )

    var rm_path = "remove_me.txt"
    try:
        os.remove(rm_path)
    except:
        pass
    var rf = FileHandle(rm_path, "w")
    rf.write("x")
    rf.close()
    check(os.path.exists(rm_path), "remove target created", failures)
    os.remove(rm_path)
    check(not os.path.exists(rm_path), "remove deletes the file", failures)

    if failures == 0:
        print("wasi_features: ALL PASS")
    else:
        print("wasi_features:", failures, "FAILURES")
        exit(1)
