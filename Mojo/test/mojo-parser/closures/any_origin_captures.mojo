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
# RUN: %parse-mojo-isolated -verify-diagnostics %s

# MOCO-5375: A closure hides the origins of its captures, so a captured
# AnyOrigin does not keep the value it refers to alive.


def use(f: Some[def() -> Int]) -> Int:
    return f()


def testImmCapture(p: Pointer[Int, MutAnyOrigin]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields' to the closure's 'def' or to an enclosing function if the lifetime is managed explicitly}}
    # expected-warning @below {{capture 'p' of type 'Pointer[Int, MutAnyOrigin]' has AnyOrigin, which the closure does not track; the value it refers to can be destroyed before the closure runs}}
    def read() {p} -> Int:
        return p[]

    return use(read)


def testVarCapture(p: Pointer[Int, MutAnyOrigin]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 'p' of type 'Pointer[Int, MutAnyOrigin]' has AnyOrigin}}
    def read() {var p} -> Int:
        return p[]

    return use(read)


def testMoveCapture(p: Pointer[Int, MutAnyOrigin]) -> Int:
    var q = p
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 'q' of type 'Pointer[Int, MutAnyOrigin]' has AnyOrigin}}
    def read() {q^} -> Int:
        return q[]

    return use(read)


def testMutCapture(p: Pointer[Int, MutAnyOrigin]) -> Int:
    var q = p
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 'q' of type 'Pointer[Int, MutAnyOrigin]' has AnyOrigin}}
    def read() {mut q} -> Int:
        return q[]

    return use(read)


# COM: An implicit capture is reported at its first use in the body.
def testDefaultCapture(p: Pointer[Int, MutAnyOrigin]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    def read() {var} -> Int:
        # expected-warning @below {{capture 'p' of type 'Pointer[Int, MutAnyOrigin]' has AnyOrigin}}
        return p[]

    return use(read)


def testImmAnyOrigin(p: Pointer[Int, AnyOrigin[mut=False]]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 'p' of type 'Pointer[Int, ImmUnsafeAnyOrigin]' has AnyOrigin}}
    def read() {p} -> Int:
        return p[]

    return use(read)


def testSpanCapture(s: Span[Int, MutAnyOrigin]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 's' of type 'Span[Int, MutAnyOrigin]' has AnyOrigin}}
    def read() {s} -> Int:
        _ = s
        return 0

    return use(read)


def testLambda(p: Pointer[Int, MutAnyOrigin]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 'p' of type 'Pointer[Int, MutAnyOrigin]' has AnyOrigin}}
    return use(lambda () {p} -> Int: p[])


# COM: Only the capture with an AnyOrigin is reported.
def testOtherCapturesAreSilent(p: Pointer[Int, MutAnyOrigin]) -> Int:
    var n = 1
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 'p' of type}}
    def read() {n, p} -> Int:
        return p[] + n

    return use(read)


def testTrackedOriginIsSilent() -> Int:
    var n = 1
    var p = Pointer(to=n)

    def read() {p} -> Int:
        return p[]

    return use(read)


def testDecoratedClosure(p: Pointer[Int, MutAnyOrigin]) -> Int:
    @__allow_legacy_any_origin_fields
    def read() {p} -> Int:
        return p[]

    return use(read)


# COM: The decorator covers every closure inside the function, at any depth.
@__allow_legacy_any_origin_fields
def testDecoratedEnclosingFunction(p: Pointer[Int, MutAnyOrigin]) -> Int:
    def read() {p} -> Int:
        return p[]

    def outer() {var} -> Int:
        def inner() {var} -> Int:
            return p[]

        return inner()

    return use(read) + use(outer) + use(lambda () {p} -> Int: p[])


struct Holder:
    @__allow_legacy_any_origin_fields
    def method(self, p: Pointer[Int, MutAnyOrigin]) -> Int:
        return use(lambda () {p} -> Int: p[])


# COM: The decorator does not reach closures in other functions.
def testSiblingOfDecoratedFunction(p: Pointer[Int, MutAnyOrigin]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    # expected-warning @below {{capture 'p' of type}}
    def read() {p} -> Int:
        return p[]

    return use(read)


# COM: A nested closure's capture is also a capture of the closure around it.
def testNestedClosures(p: Pointer[Int, MutAnyOrigin]) -> Int:
    # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
    def outer() {var} -> Int:
        # expected-note @below {{add '@__allow_legacy_any_origin_fields'}}
        def inner() {var} -> Int:
            # expected-warning @below {{capture 'p' of type}}
            return p[]

        return inner()

    return use(outer)
