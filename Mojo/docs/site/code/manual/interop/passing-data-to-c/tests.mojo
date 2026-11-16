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
# tests.mojo
# Tests for interop/passing-data-to-c.mdx. Banners follow the page's
# headings. Every call goes to the C standard library, as on the page.
#
# Page claims, and where each is verified:
#    1. `frexp()` returns the mantissa and writes the exponent through a
#       C-style out argument                          -> test_frexp
#    2. `div()` returns a struct by value              -> test_div
#    3. `clock_gettime()` fills a `timespec` you own   -> test_clock_gettime
#    4. `fopen()`/`fgets()`/`fclose()` read a line through a handle
#                                                     -> test_file_handle
#    5. `malloc()` memory is usable through `unsafe_offset=` and freed with
#       `free()`                                       -> test_malloc
#    6. `__exit__()` frees the memory when the `with` block ends, even when
#       the block raises                              -> test_cbuffer
#    7. `getentropy()` fills a Mojo buffer, counting bytes
#                                                     -> test_getentropy
#    8. `strdup()` -> `putenv()` -> `getenv()` sets and reads a variable
#                                                     -> test_environment
#    9. `gethostname()` writes into a Mojo `Array`; passing one byte less keeps
#       the last byte zero                            -> test_gethostname
#   10. `getcwd()` returns a pointer into your buffer  -> test_getcwd
#   11. `bsearch()` finds a key, and returns null when it finds nothing
#                                                     -> test_bsearch
#   12. The comparison function returns -1, 0, or 1   -> test_compare_ports
#   13. `get_errno()` reports why `fopen()` failed     -> test_errno
#   14. A wrapper raises when `clock_gettime()` fails  -> test_clock_nanos
#
# Not tested:
#   - Passing a bare `17` instead of `c_int(17)`, and the "Check your
#     declarations" mismatches: undefined behavior that varies by platform.
#   - A declaration that conflicts with the standard library's: a compile
#     error.
#   - `putenv()` keeping its pointer, `getenv()` memory changing later, and
#     stored callbacks: lifetime rules with no stable result to assert.
#   - A write through a pointer to an immutable argument, and `bsearch()` on
#     an unsorted array: undefined behavior.
#   - `free()` through `external_call` next to `input()` or
#     `subprocess.run()`: a compile error.
#   - The `qsort()` link: functions/lambda tests cover that example.
from std.ffi import (
    CStringSpan,
    c_char,
    c_double,
    c_int,
    c_long,
    c_size_t,
    external_call,
    get_errno,
)
from std.ffi import ErrNo
from std.sys import size_of
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

comptime FILE = Optional[MutOpaquePointer[MutUntrackedOrigin]]


# --- C-style out arguments ---


def test_frexp() raises:
    # double frexp(double x, int *exp);
    var exponent: c_int = 0
    var mantissa = external_call["frexp", c_double](
        c_double(12.0), Pointer(to=exponent)
    )
    assert_equal(mantissa, 0.75)
    assert_equal(exponent, 4)


# --- Structs ---


@fieldwise_init
struct DivT(RegisterPassable):
    var quot: c_int
    var rem: c_int


def test_div() raises:
    var r = external_call["div", DivT](c_int(17), c_int(5))
    assert_equal(r.quot, 3)
    assert_equal(r.rem, 2)


comptime CLOCK_REALTIME = c_int(0)


@fieldwise_init
struct Timespec:
    var tv_sec: Int64  # time_t
    var tv_nsec: c_long


def test_clock_gettime() raises:
    var now = Timespec(tv_sec=0, tv_nsec=0)
    # int clock_gettime(clockid_t clock, struct timespec *tp);
    var status = external_call["clock_gettime", c_int](
        CLOCK_REALTIME, Pointer(to=now)
    )
    assert_equal(status, 0)
    assert_true(now.tv_sec > 1_700_000_000)  # after November 2023
    assert_true(now.tv_nsec >= 0 and now.tv_nsec < 1_000_000_000)


# --- Opaque handles ---


def test_file_handle() raises:
    # FILE *fopen(const char *path, const char *mode);
    var file = external_call["fopen", FILE](
        "/etc/hosts".as_c_string_span(), "r".as_c_string_span()
    )
    assert_true(Bool(file))
    var line = Array[c_char, 256](fill=0)
    # char *fgets(char *buf, int size, FILE *stream);
    var read = external_call[
        "fgets", Optional[Pointer[c_char, MutUntrackedOrigin]]
    ](line.unsafe_ptr(), c_int(len(line)), file)
    assert_true(Bool(read))
    assert_true(external_call["strlen", c_size_t](line.unsafe_ptr()) > 0)
    var text = CStringSpan(unsafe_from_ptr=line.unsafe_ptr())
    assert_true(String(from_utf8=text.as_bytes()).byte_length() > 0)
    # int fclose(FILE *stream);
    assert_equal(external_call["fclose", c_int](file), 0)


# --- C memory ---


def test_malloc() raises:
    # void *malloc(size_t size);
    var block = external_call[
        "malloc", Optional[Pointer[UInt8, MutUntrackedOrigin]]
    ](c_size_t(64))
    assert_true(Bool(block))
    var bytes = block.value()
    bytes[unsafe_offset=0] = 42
    bytes[unsafe_offset=63] = 7
    assert_equal(bytes[unsafe_offset=0], 42)
    assert_equal(bytes[unsafe_offset=63], 7)
    # void free(void *ptr);
    external_call["free", NoneType](bytes)


# The page's CBuffer, plus a counter so the test can see `__exit__()` run.
struct CBuffer:
    var ptr: Pointer[UInt8, MutUntrackedOrigin]
    var frees: Pointer[Int, MutUntrackedOrigin]

    def __init__(
        out self, size: c_size_t, frees: Pointer[Int, MutUntrackedOrigin]
    ) raises:
        var block = external_call[
            "malloc", Optional[Pointer[UInt8, MutUntrackedOrigin]]
        ](size)
        if not block:
            raise Error("malloc failed")
        self.ptr = block.value()
        self.frees = frees

    def __enter__(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self.ptr

    def __exit__(self):
        external_call["free", NoneType](self.ptr)
        self.frees[] += 1


def test_cbuffer() raises:
    var frees = 0
    var counter = Pointer(to=frees).unsafe_origin_cast[MutUntrackedOrigin]()
    with CBuffer(c_size_t(1024), counter) as buf:
        buf[unsafe_offset=0] = 42
        assert_equal(buf[unsafe_offset=0], 42)
    assert_equal(frees, 1)
    try:
        with CBuffer(c_size_t(1024), counter) as buf:
            buf[unsafe_offset=0] = 42
            raise Error("stop")
    except:
        pass
    assert_equal(frees, 2)


# --- Buffers ---


def test_getentropy() raises:
    var key = Array[UInt8, 16](fill=0)
    # int getentropy(void *buf, size_t len);
    var status = external_call["getentropy", c_int](
        key.unsafe_ptr(), c_size_t(len(key))
    )
    assert_equal(status, 0)
    var nonzero = 0
    for byte in key:
        if byte != 0:
            nonzero += 1
    assert_true(nonzero > 0)  # 16 zero bytes has odds of 1 in 2**128


# --- Strings ---


def test_environment() raises:
    var setting = String("OMP_NUM_THREADS=4")
    # char *strdup(const char *s);
    var copy = external_call[
        "strdup", Optional[Pointer[c_char, MutUntrackedOrigin]]
    ](setting.as_c_string_span())
    assert_true(Bool(copy))
    # int putenv(char *string);
    assert_equal(external_call["putenv", c_int](copy), 0)
    # char *getenv(const char *name);
    var value = external_call[
        "getenv", Optional[Pointer[c_char, ImmUntrackedOrigin]]
    ]("OMP_NUM_THREADS".as_c_string_span())
    assert_true(Bool(value))
    var text = CStringSpan(unsafe_from_ptr=value.value())
    assert_equal(String(from_utf8=text.as_bytes()), "4")


def test_gethostname() raises:
    var name = Array[c_char, 256](fill=0)
    # int gethostname(char *name, size_t len);
    var status = external_call["gethostname", c_int](
        name.unsafe_ptr(), c_size_t(len(name) - 1)
    )
    assert_equal(status, 0)
    assert_true(
        String(unsafe_from_utf8_ptr=name.unsafe_ptr()).byte_length() > 0
    )
    assert_equal(name[len(name) - 1], 0)


def test_getcwd() raises:
    var buf = Array[c_char, 1024](fill=0)
    # char *getcwd(char *buf, size_t size);
    var path = external_call[
        "getcwd", Optional[Pointer[c_char, origin_of(buf)]]
    ](buf.unsafe_ptr(), c_size_t(len(buf)))
    assert_true(Bool(path))
    assert_true(path.value() == buf.unsafe_ptr())
    assert_true(String(unsafe_from_utf8_ptr=path.value()).startswith("/"))


# --- Callbacks ---


def compare_ports(
    key: ImmOpaquePointer[ImmUntrackedOrigin],
    item: ImmOpaquePointer[ImmUntrackedOrigin],
) abi("C") -> c_int:
    var wanted = key.unsafe_bitcast[UInt16]()[]
    var entry = item.unsafe_bitcast[UInt16]()[]
    return c_int(-1) if wanted < entry else c_int(wanted > entry)


def lookup(port: UInt16) -> Bool:
    var well_known: List[UInt16] = [22, 25, 53, 80, 123, 143, 443, 993]
    var key = port
    # void *bsearch(const void *key, const void *base, size_t count,
    #               size_t size, int (*compare)(const void *, const void *));
    var found = external_call[
        "bsearch", Optional[Pointer[UInt16, ImmUntrackedOrigin]]
    ](
        Pointer(to=key),
        well_known.unsafe_ptr(),
        c_size_t(len(well_known)),
        c_size_t(size_of[UInt16]()),
        compare_ports,
    )
    return Bool(found)


def test_bsearch() raises:
    assert_true(lookup(443))
    assert_true(lookup(22))
    assert_true(lookup(993))
    assert_false(lookup(444))
    assert_false(lookup(0))


def compare(a: UInt16, b: UInt16) -> c_int:
    return compare_ports(
        Pointer(to=a)
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[ImmUntrackedOrigin](),
        Pointer(to=b)
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[ImmUntrackedOrigin](),
    )


def test_compare_ports() raises:
    assert_equal(compare(0, 65535), -1)
    assert_equal(compare(65535, 0), 1)
    assert_equal(compare(443, 443), 0)


# --- Errors ---


def test_errno() raises:
    # FILE *fopen(const char *path, const char *mode);
    var file = external_call["fopen", FILE](
        "/no/such/file".as_c_string_span(), "r".as_c_string_span()
    )
    assert_false(Bool(file))
    assert_true(get_errno() == ErrNo.ENOENT)


def clock_nanos(clock: c_int) raises -> Int64:
    var now = Timespec(tv_sec=0, tv_nsec=0)
    # int clock_gettime(clockid_t clock, struct timespec *tp);
    if external_call["clock_gettime", c_int](clock, Pointer(to=now)) != 0:
        raise Error("clock_gettime failed: ", get_errno())
    return now.tv_sec * 1_000_000_000 + Int64(now.tv_nsec)


def test_clock_nanos() raises:
    assert_true(clock_nanos(CLOCK_REALTIME) > 0)
    with assert_raises(contains="clock_gettime failed"):
        _ = clock_nanos(c_int(9999))


def main() raises:
    test_frexp()
    test_div()
    test_clock_gettime()
    test_file_handle()
    test_malloc()
    test_cbuffer()
    test_getentropy()
    test_environment()
    test_gethostname()
    test_getcwd()
    test_bsearch()
    test_compare_ports()
    test_errno()
    test_clock_nanos()
