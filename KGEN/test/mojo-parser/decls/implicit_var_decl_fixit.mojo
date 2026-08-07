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

# Tests for the fixit on the implicit-declaration warning. Uses the JSON
# diagnostic format to verify the exact fixit position, 'var ' immediately
# before the declared name, and the three shapes that deliberately offer no
# fixit: a tuple element, because one 'var' covers the whole target; a site
# inside a nested block, because 'var' there would be scoped to the block; and a
# walrus target, because 'var' and 'ref' on one are being removed from the
# language.
# Related to MOCO-3182.

# RUN: %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s


def one() -> Int:
    return 1


def truthy() -> Bool:
    return True


def use(x: Int):
    pass


def simple():
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'x' is deprecated; add 'var' before the name"
    x = 1
    use(x)


# The insertion goes before the name, not before the type annotation.
def annotated():
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'y' is deprecated; add 'var' before the name"
    y: Int = 5
    use(y)


# A walrus target gets no fixit even at function-body level, where inserting
# 'var' at the name would have been the equivalent edit for a plain assignment.
def walrus_in_call():
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'v' is deprecated; declare it with 'var' in the function body"
    use(v := one())
    use(v)


# A walrus in a condition is a nested site as well, and reads the same.
def walrus():
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'z' is deprecated; declare it with 'var' in the function body"
    if z := truthy():
        use(1)


# No fixit inside a nested block either: 'var e' there would be scoped to the
# block, while the declaration is scoped to the function.
def nested_block(c: Bool):
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'e' is deprecated; declare it with 'var' in the function body"
    if c:
        e = 1
    use(1)


# A short-circuit operand is reachable only by a walrus, and gets a block of its
# own besides.
def and_operand(c: Bool) -> Bool:
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'w' is deprecated; declare it with 'var' in the function body"
    return c and (w := truthy())


# Both targets of a chain get their own fixit, and 'var a = var b = 1' compiles.
def chained():
    # CHECK: "fixIts":[{"end":{"column":9,"line":[[#@LINE+4]]},"start":{"column":9,"line":[[#@LINE+4]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'b' is deprecated; add 'var' before the name"
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'a' is deprecated; add 'var' before the name"
    a = b = 1
    use(a)
    use(b)


# No fixit on a tuple element: 'var c, var d = ...' is a nested pattern, which
# the compiler rejects.
def tuple_target():
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'c' is deprecated; add 'var' before the assignment target"
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'd' is deprecated; add 'var' before the assignment target"
    c, d = Tuple(1, 2)
    use(c)
    use(d)


# A binder one level up in the target rules out the outer 'var' for the names
# nested inside it, so they take the hoisted-declaration message and no fixit.
def nested_tuple_sibling():
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'f' is deprecated; declare it with 'var' in the function body"
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'g' is deprecated; declare it with 'var' in the function body"
    (f, g), var h = Tuple(Tuple(1, 2), 3)
    use(f)
    use(g)
    use(h)


def shadow():
    # CHECK: "fixIts":[{"end":{"column":5,"line":[[#@LINE+2]]},"start":{"column":5,"line":[[#@LINE+2]]},"text":"var "}]
    # CHECK-SAME: "message":"implicit declaration of 'one' is deprecated; add 'var' before the name"
    one = 2
    use(one)


# A shadowing site takes whichever base message its position calls for, so pin
# the other two positions here too.
def nested_shadow(c: Bool):
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'one' is deprecated; declare it with 'var' in the function body"
    if c:
        one = 2
    use(1)


def tuple_shadow():
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'one' is deprecated; add 'var' before the assignment target"
    # CHECK: "fixIts":[]
    # CHECK-SAME: "message":"implicit declaration of 'x' is deprecated; add 'var' before the assignment target"
    one, x = Tuple(1, 2)
    use(one)
    use(x)
