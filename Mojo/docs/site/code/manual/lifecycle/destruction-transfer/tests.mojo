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
# Tests for destruction-transfer.mdx code examples and prose claims.
#
# This page is about the transfer sigil's relationship to destruction. It sits
# alongside death.mdx, which owns ASAP destruction itself. Claims that page
# already owns are NOT retested here; see
# ../death/tests.mojo for:
#   - ASAP ordering, including within an expression
#   - field ASAP inside __deinit__
#   - `_ = value` as a lifetime extension, and the RAII guard pair
#   - origins keeping an owner alive, and a severed origin needing an extension
#
# The page demonstrates destruction by printing from a deinitializer. Printing
# can't be asserted, so the types here record into a list instead and the tests
# assert the recorded order. Note that the recorder measures APPEND order: a
# test that builds its event string from a field read will see that read count
# as the value's last use, and the deinitializer will run before the append.
# Record first, read the field last.
#
# Page claims, and where each is verified:
#   1. A discard is a use, not the end of a value: `_ = x` and `_ = x^` are
#      indistinguishable, and the value dies at its real last use.
#      -> test_discard_sigil_is_inert
#   2. When the discard is the value's last use, the value dies right after it.
#      -> test_discard_as_last_use
#
# Covered by other suites: a consumer ending a value (a `var` binding in
# ../trivial/tests.mojo, a `deinit self` method in ../explicit-destroy/).
#
# Not tested here, because these are compile-time diagnostics and a runtime
# binary cannot observe them. They need a lit test with expected-error and
# expected-warning directives:
#   - "Sigils that may confuse": the trivial-scalar, already-owned, and `x^^`
#     warnings, each of which carries a remove fix-it
#   - "When the sigil is rejected": transferring out of a `read` or `ref`
#     binding, and a `mut` binding left un-restored at return
#   - a field transferred out of an owned value, leaving it incomplete
#   - `x.__deinit__()` destroying a copy. The behavior is real and the page
#     documents it, but demonstrating it REQUIRES emitting the
#     "destroys a copy of the value" warning, and CI compiles this target
#     with -Werror. It has to live in a lit test with an expected-warning
#     directive, not here.
from std.testing import assert_equal

# --- Recording helper: an origin-erased handle to a list of events ---

comptime Recorder = Pointer[List[String], MutUntrackedOrigin]


def recorder(mut events: List[String]) -> Recorder:
    return Pointer(to=events).unsafe_origin_cast[MutUntrackedOrigin]()


def record(into: Recorder, var event: String):
    into[].append(event^)


@fieldwise_init
struct Guard(Movable):
    var name: String
    var events: Recorder

    def __deinit__(deinit self):
        record(self.events, String("released ", self.name))


# --- A discard is a use, not the end of a value ---


def test_discard_sigil_is_inert() raises:
    # `_ = x` and `_ = x^` are indistinguishable: same legality, same order.
    var with_sigil = List[String]()
    var a = recorder(with_sigil)
    var g1 = Guard("x", a)
    _ = g1^
    record(a, "after")
    assert_equal(g1.name, "x")

    var without = List[String]()
    var b = recorder(without)
    var g2 = Guard("x", b)
    _ = g2
    record(b, "after")
    assert_equal(g2.name, "x")

    assert_equal(len(with_sigil), len(without))
    for i in range(len(with_sigil)):
        assert_equal(with_sigil[i], without[i])
    # The later read is the real last use, so the value outlives the discard.
    assert_equal(with_sigil[0], "after")
    assert_equal(with_sigil[1], "released x")


def test_discard_as_last_use() raises:
    # When the discard is the last use, the value dies right after it.
    var events = List[String]()
    var log = recorder(events)
    var g = Guard("last", log)
    _ = g^
    record(log, "after")

    assert_equal(events[0], "released last")
    assert_equal(events[1], "after")


def main() raises:
    test_discard_sigil_is_inert()
    test_discard_as_last_use()
