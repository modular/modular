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
# Tests for trivial.mdx code examples and prose claims.
#
# The page demonstrates destruction through a file descriptor's deinitializer.
# Closing a real descriptor can't be asserted, so `Descriptor` here records
# into a list instead and the tests assert the recorded order.
#
# Page claims, and where each is verified:
#   1. `Int`, `UInt8`, `Float64`, `Bool`, `SIMD`, and `Pointer` are
#      register-passable and trivial; `String` is neither.
#      -> test_builtin_register_types
#   2. A register type can define a deinitializer, isn't copyable until it
#      adds `Copyable`, and isn't trivial.
#      -> test_register_type_capabilities
#   3. A register value dies at its last use, by its deinitializer, and a
#      transfer hands it to the new owner (it dies once, after the new
#      owner's last use).
#      -> test_register_value_destroyed_at_last_use,
#         test_register_transfer_moves_ownership
#   4. A copyable register type can define its own copy initializer.
#      -> test_register_type_custom_copy
#   5. A register type can require explicit destruction.
#      -> test_register_type_explicit_destruction
#   6. A trivial register type copies implicitly, and copies are independent;
#      the page's documented `Sample` output.
#      -> test_sample_copies_are_independent
#   7. Trivial values compose: a struct of trivial fields can be trivial.
#      -> test_stereo_frame_is_trivial
#   8. `Pixel`, which generates its lifecycle methods from trivial fields, has
#      trivial copies and destruction but isn't implicitly copyable or
#      `TrivialRegisterPassable`.
#      -> test_pixel_trivial_without_trait
#   9. A non-trivial field, or a lifecycle method you write, makes a stage
#      non-trivial.
#      -> test_non_trivial_stages
#  10. The predicates' definitions, including that `IsTriviallyMovable` holds
#      for `String`.
#      -> test_trivially_copyable, test_trivially_movable,
#         test_trivially_deinitable
#  11. Each predicate is a `comptime` value.
#      -> test_predicates_are_comptime
#  12. `append_bytes()` writes a trivially copyable value's bytes.
#      -> test_append_bytes
#
# Compile-time claims, verified against the compiler on 2026-10-16, Mojo
# 1.2.0.dev2026101520 (d2872eb2), and recorded with the diagnostic that fires.
# CI builds this target with -Werror, so the warning can't run here either:
#   C1. A `String` field in a register type: "all members of 'RegisterPassable'
#       struct must themselves be 'RegisterPassable'". The same for a trivial
#       register type, and for a `Descriptor` field in one.
#   C2. A register type can't define a move initializer: "'RegisterPassable'
#       types must not declare explicit move constructors".
#   C3. A register type can't opt out of `Movable`: "trait 'RegisterPassable'
#       requires ancestor trait 'Movable', which is opted out with 'not'".
#   C4. A trivial register type can't define `__deinit__()` ("trivial types
#       must not declare '__deinit__' methods") or a copy initializer
#       ("trivial types must not declare explicit copy constructors").
#   C5. A trivial register type can't require explicit destruction: "trait
#       'TrivialRegisterPassable' requires ancestor trait 'Deinitable', which
#       is opted out with 'not'".
#   C6. A transfer of a trivial value warns "transfer from a value of trivial
#       register type 'Int' has no effect and can be removed", and the
#       original stays usable (the page's `43 42`).
#   C7. Reading a register value after transferring it: "use of uninitialized
#       value 'log'".
#   C8. `append_bytes()` with a `String` field fails with the assertion's
#       message, "'T' must be trivially copyable".
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_true
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)

# --- Recording helper: an origin-erased handle to a list of events ---

comptime Recorder = Pointer[List[String], MutUntrackedOrigin]


def recorder(mut events: List[String]) -> Recorder:
    return Pointer(to=events).unsafe_origin_cast[MutUntrackedOrigin]()


def record(into: Recorder, var event: String):
    into[].append(event^)


# --- Values that are just their bits ---


def test_builtin_register_types() raises:
    assert_true(conforms_to(Int, TrivialRegisterPassable))
    assert_true(conforms_to(UInt8, TrivialRegisterPassable))
    assert_true(conforms_to(Float64, TrivialRegisterPassable))
    assert_true(conforms_to(Bool, TrivialRegisterPassable))
    assert_true(conforms_to(SIMD[DType.float32, 4], TrivialRegisterPassable))
    assert_true(
        conforms_to(Pointer[Int, MutUntrackedOrigin], TrivialRegisterPassable)
    )
    assert_true(conforms_to(Int, RegisterPassable))
    assert_false(conforms_to(String, RegisterPassable))
    assert_false(conforms_to(String, TrivialRegisterPassable))


# --- Register types ---


struct Descriptor(RegisterPassable):
    """Mocks the page's `Descriptor`: records instead of closing a file."""

    var fd: Int32
    var events: Recorder

    def __init__(out self, fd: Int32, events: Recorder):
        self.fd = fd
        self.events = events

    def use(self, what: String):
        record(self.events, String(what, " ", self.fd))

    def __deinit__(deinit self):
        record(self.events, String("close ", self.fd))


def test_register_type_capabilities() raises:
    assert_true(conforms_to(Descriptor, RegisterPassable))
    assert_true(conforms_to(Descriptor, Movable))
    assert_false(conforms_to(Descriptor, Copyable))
    assert_false(conforms_to(Descriptor, TrivialRegisterPassable))
    assert_false(IsTriviallyDeinitable[Descriptor])


def test_register_value_destroyed_at_last_use() raises:
    var events = List[String]()
    var log = recorder(events)

    var descriptor = Descriptor(3, log)
    descriptor.use("using")  # last use
    record(log, "after last use")

    var expected = ["using 3", "close 3", "after last use"]
    assert_equal(len(events), len(expected))
    for i in range(len(expected)):
        assert_equal(events[i], expected[i])


def test_register_transfer_moves_ownership() raises:
    var events = List[String]()
    var log = recorder(events)

    var log_fd = Descriptor(4, log)
    var keeper = log_fd^
    record(log, "transferred")
    keeper.use("keeper holds")  # keeper's last use

    # One value, destroyed once, after the new owner's last use.
    var expected = ["transferred", "keeper holds 4", "close 4"]
    assert_equal(len(events), len(expected))
    for i in range(len(expected)):
        assert_equal(events[i], expected[i])


struct SharedDescriptor(Copyable, RegisterPassable):
    """A copy means opening a second descriptor, modeled as a new number."""

    var fd: Int32

    def __init__(out self, fd: Int32):
        self.fd = fd

    def __init__(out self, *, copy: Self):
        self.fd = copy.fd + 100


def test_register_type_custom_copy() raises:
    var original = SharedDescriptor(3)
    var duplicate = original.copy()
    assert_equal(duplicate.fd, 103)
    assert_equal(original.fd, 3)


struct Lease(
    not Deinitable else "call 'release()' to return the lease", RegisterPassable
):
    var id: Int32
    var events: Recorder

    def __init__(out self, id: Int32, events: Recorder):
        self.id = id
        self.events = events

    def release(deinit self):
        record(self.events, String("released ", self.id))


def test_register_type_explicit_destruction() raises:
    assert_true(conforms_to(Lease, RegisterPassable))
    assert_false(conforms_to(Lease, Deinitable))

    var events = List[String]()
    var log = recorder(events)
    var lease = Lease(7, log)
    lease^.release()

    assert_equal(len(events), 1)
    assert_equal(events[0], "released 7")


# --- Trivial register types ---


@fieldwise_init
struct Sample(TrivialRegisterPassable, Writable):
    var channel: UInt8
    var value: Float32


def amplify(var sample: Sample, gain: Float32) -> Sample:
    sample.value *= gain  # changes this function's copy only
    return sample


def test_sample_copies_are_independent() raises:
    assert_true(conforms_to(Sample, ImplicitlyCopyable))

    var raw = Sample(channel=2, value=0.5)
    var edited = raw  # implicit copy
    edited.value = 0.25
    var louder = amplify(raw, 2.0)

    assert_equal(String(raw), "Sample(channel=2, value=0.5)")
    assert_equal(String(edited), "Sample(channel=2, value=0.25)")
    assert_equal(String(louder), "Sample(channel=2, value=1.0)")


@fieldwise_init
struct StereoFrame(TrivialRegisterPassable):
    var left: Sample
    var right: Sample


def test_stereo_frame_is_trivial() raises:
    assert_true(conforms_to(StereoFrame, TrivialRegisterPassable))

    var frame = StereoFrame(
        left=Sample(channel=0, value=0.5), right=Sample(channel=1, value=1.0)
    )
    var saved = frame  # implicit copy
    saved.left.value = 0.0
    assert_equal(frame.left.value, 0.5)
    assert_equal(saved.right.value, 1.0)


# --- Trivial lifecycles without the trait ---


@fieldwise_init
struct Pixel(Copyable):
    var red: UInt8
    var green: UInt8
    var blue: UInt8


def test_pixel_trivial_without_trait() raises:
    assert_true(IsTriviallyCopyable[Pixel])
    assert_true(IsTriviallyDeinitable[Pixel])
    assert_false(conforms_to(Pixel, ImplicitlyCopyable))
    assert_false(conforms_to(Pixel, TrivialRegisterPassable))

    var pixel = Pixel(red=255, green=128, blue=0)
    var backup = pixel.copy()
    assert_equal(backup.green, 128)


@fieldwise_init
struct Label(Copyable):
    var text: String  # a non-trivial field


@fieldwise_init
struct TracedPixel(Copyable):
    var red: UInt8
    var events: Recorder

    def __deinit__(deinit self):
        record(self.events, "destroyed")  # a lifecycle method you write


def test_non_trivial_stages() raises:
    assert_false(IsTriviallyCopyable[Label])
    assert_false(IsTriviallyDeinitable[Label])
    assert_false(IsTriviallyDeinitable[TracedPixel])
    # Writing only a deinitializer leaves the generated copy trivial.
    assert_true(IsTriviallyCopyable[TracedPixel])


# --- Check a lifecycle in generic code ---


comptime Trivial = Int


@fieldwise_init
struct SideEffects(Copyable, Movable):
    """Custom lifecycle methods, so no operation is trivial."""

    var value: Int

    def __init__(out self, *, copy: Self):
        self.value = copy.value + 1  # a side effect of copying

    def __init__(out self, *, deinit move: Self):
        self.value = move.value

    def __deinit__(deinit self):
        _ = self.value  # a deinitializer body, so deinit is not a no-op


def test_trivially_copyable() raises:
    assert_true(IsTriviallyCopyable[Trivial])
    assert_true(IsTriviallyCopyable[SIMD[DType.float32, 4]])
    # `String` is `Copyable`, but copying owns heap storage.
    assert_false(IsTriviallyCopyable[String])
    assert_false(IsTriviallyCopyable[SideEffects])


def test_trivially_movable() raises:
    assert_true(IsTriviallyMovable[Trivial])
    assert_true(IsTriviallyMovable[SIMD[DType.float32, 4]])
    # The page's claim: moving is trivial for most types, `String` included.
    assert_true(IsTriviallyMovable[String])
    assert_false(IsTriviallyMovable[SideEffects])


def test_trivially_deinitable() raises:
    assert_true(IsTriviallyDeinitable[Trivial])
    assert_true(IsTriviallyDeinitable[SIMD[DType.float32, 4]])
    # `String` releases heap storage, so its deinitializer is not a no-op.
    assert_false(IsTriviallyDeinitable[String])
    assert_false(IsTriviallyDeinitable[SideEffects])


def test_predicates_are_comptime() raises:
    comptime assert IsTriviallyCopyable[Trivial]
    comptime assert not IsTriviallyDeinitable[String]

    var branch_taken: Bool
    comptime if IsTriviallyMovable[Trivial]:
        branch_taken = True
    else:
        branch_taken = False
    assert_true(branch_taken)


@fieldwise_init
struct Header(Copyable):
    var magic: UInt32
    var count: UInt32


def append_bytes[T: Copyable](mut buffer: List[Byte], value: T):
    comptime assert IsTriviallyCopyable[T], "'T' must be trivially copyable"
    var bytes = Pointer(to=value).unsafe_bitcast[Byte]()
    for i in range(size_of[T]()):
        buffer.append(bytes.unsafe_offset(i)[])


def test_append_bytes() raises:
    var buffer = List[Byte]()
    append_bytes(buffer, Header(magic=0x4D4F4A4F, count=3))
    assert_equal(len(buffer), 8)
    # Little-endian bytes of the two fields, in declaration order.
    var expected: List[Byte] = [0x4F, 0x4A, 0x4F, 0x4D, 3, 0, 0, 0]
    for i in range(len(expected)):
        assert_equal(buffer[i], expected[i])


def main() raises:
    test_builtin_register_types()
    test_register_type_capabilities()
    test_register_value_destroyed_at_last_use()
    test_register_transfer_moves_ownership()
    test_register_type_custom_copy()
    test_register_type_explicit_destruction()
    test_sample_copies_are_independent()
    test_stereo_frame_is_trivial()
    test_pixel_trivial_without_trait()
    test_non_trivial_stages()
    test_trivially_copyable()
    test_trivially_movable()
    test_trivially_deinitable()
    test_predicates_are_comptime()
    test_append_bytes()
