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

# Tests for the deprecation warning on an implicitly declared variable, and for
# the binding forms that must stay silent because they already spell out how
# they bind.
# Related to MOCO-3182.

# RUN: %parse-mojo-isolated %s -verify-diagnostics -o /dev/null


struct ExampleCM(ImplicitlyCopyable):
    def __enter__(self) -> Int:
        return 1

    def __exit__(self):
        pass


def one() -> Int:
    return 1


def truthy() -> Bool:
    return True


def use(x: Int):
    pass


def use_bool(x: Bool):
    pass


def simple():
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; add 'var' before the name}}
    x = 1
    use(x)


def annotated():
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; add 'var' before the name}}
    x: Int = 5
    use(x)


# An annotation with no initializer declares just as much as an assignment does.
def annotated_no_initializer():
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; add 'var' before the name}}
    x: Int
    x = 5
    use(x)


# Only the declaration warns; later assignments to the same name are stores.
def reassigned():
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; add 'var' before the name}}
    x = 1
    x = 2
    use(x)


# A single 'var' covers a whole tuple target, so each name gets the target
# message rather than the name message a lone assignment gets.
def tuple_target():
    # expected-warning @+2 {{implicit declaration of 'a' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+1 {{implicit declaration of 'b' is deprecated; add 'var' before the assignment target}}
    a, b = Tuple(1, 2)
    use(a)
    use(b)


# Only the fresh element warns when the target mixes declared and fresh names.
def tuple_mixed():
    var a = 0
    # expected-warning @+1 {{implicit declaration of 'b' is deprecated; add 'var' before the assignment target}}
    a, b = Tuple(1, 2)
    use(a)
    use(b)


# A binder on a sibling rules out prefixing the whole target with 'var', so the
# fresh element is told to declare itself separately instead.
def tuple_sibling_ref(t: Tuple[Int, Int]):
    # expected-warning @+1 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    a, ref b = t
    use(a)
    use(b)


def tuple_sibling_var():
    # expected-warning @+1 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    a, var b = Tuple(1, 2)
    use(a)
    use(b)


# The spelling the warning above asks for, and the one the stdlib uses: a
# separate declaration, then a plain assignment that binds the sibling.
def tuple_sibling_ref_migrated(t: Tuple[Int, Int]):
    var a: Int
    a, ref b = t
    use(a)
    use(b)


# With no binder anywhere in the target, a nested one still takes a single outer
# 'var', so every name keeps the target message.
def nested_tuple_target_no_binder():
    # expected-warning @+3 {{implicit declaration of 'a' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+2 {{implicit declaration of 'b' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+1 {{implicit declaration of 'c' is deprecated; add 'var' before the assignment target}}
    (a, b), c = Tuple(Tuple(1, 2), 3)
    use(a)
    use(b)
    use(c)


def nested_tuple_target_no_binder_migrated():
    var (a, b), c = Tuple(Tuple(1, 2), 3)
    use(a)
    use(b)
    use(c)


# The binder need not be a sibling of the diagnosed name: an outer 'var' has to
# prefix the whole target, so a binder anywhere in it rules that spelling out for
# every fresh name, at any depth.
def nested_tuple_sibling_var():
    # expected-warning @+2 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+1 {{implicit declaration of 'b' is deprecated; declare it with 'var' in the function body}}
    (a, b), var c = Tuple(Tuple(1, 2), 3)
    use(a)
    use(b)
    use(c)


# TODO(KGEN-XXXX): the migrated form of the case above belongs here, spelling
# the two names as `var a: Int` / `var b: Int`. It is omitted because an inner
# tuple that fully resolves, beside a sibling that does not, strands an owning
# `RCRef<TupleDLValue>` in the parser's persistent arena, and LeakSanitizer
# reports it. The unmigrated form above is unaffected: its names are fresh, so
# the inner tuple never resolves and no `TupleDLValue` is formed.


def nested_tuple_sibling_ref(t: Tuple[Tuple[Int, Int], Int]):
    # expected-warning @+2 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+1 {{implicit declaration of 'b' is deprecated; declare it with 'var' in the function body}}
    (a, b), ref c = t
    use(a)
    use(b)
    use(c)


def nested_tuple_sibling_deep():
    # expected-warning @+3 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+2 {{implicit declaration of 'b' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+1 {{implicit declaration of 'c' is deprecated; declare it with 'var' in the function body}}
    ((a, b), c), var d = Tuple(Tuple(Tuple(1, 2), 3), 4)
    use(a)
    use(b)
    use(c)
    use(d)


# Each target of a chain answers for itself, so the one without a binder still
# takes the outer 'var'.
def chain_tuple_sibling():
    # expected-warning @+4 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+3 {{implicit declaration of 'b' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+2 {{implicit declaration of 'p' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+1 {{implicit declaration of 'q' is deprecated; add 'var' before the assignment target}}
    p, q = (a, b), var c = Tuple(Tuple(1, 2), 3)
    use(a)
    use(b)
    use(c)
    use(q)


# TODO(KGEN-XXXX): the migrated form of the chain above belongs here; omitted
# for the same arena leak.


# Each target of a chain declares, and 'var' on each is a valid spelling.
def chained():
    # expected-warning @+2 {{implicit declaration of 'a' is deprecated; add 'var' before the name}}
    # expected-warning @+1 {{implicit declaration of 'b' is deprecated; add 'var' before the name}}
    a = b = 1
    use(a)
    use(b)


def chained_migrated():
    var a = var b = 1
    use(a)
    use(b)


# A chain of tuple targets has one 'var' per target, so the same message points
# at two different targets in the one statement.
def chain_tuple():
    # expected-warning @+4 {{implicit declaration of 'a' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+3 {{implicit declaration of 'b' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+2 {{implicit declaration of 'c' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+1 {{implicit declaration of 'd' is deprecated; add 'var' before the assignment target}}
    a, b = c, d = Tuple(1, 2)
    use(a)
    use(b)
    use(c)
    use(d)


def chain_tuple_migrated():
    var a, b = var c, d = Tuple(1, 2)
    use(a)
    use(b)
    use(c)
    use(d)


# A walrus target takes the hoisting message wherever it appears: 'var' and
# 'ref' on a walrus target are being removed from the language, so neither is an
# edit to ask for.
def walrus_condition():
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    if x := truthy():
        use_bool(x)


# The hoisted form that message asks for: the walrus then assigns the
# declaration instead of introducing one.
def walrus_condition_hoisted():
    var x: Bool
    if x := truthy():
        use_bool(x)
    use_bool(x)


# A call argument is not a block of its own, so this one turns on the walrus
# alone rather than on where it sits.
def walrus_in_call():
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    use(x := 1)


def walrus_in_call_hoisted():
    var x: Int
    use(x := 1)


# One 'var' covers a whole tuple target, but a walrus target takes none, so each
# element hoists instead of naming the target.
def walrus_tuple_target():
    # expected-warning @+2 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+1 {{implicit declaration of 'b' is deprecated; declare it with 'var' in the function body}}
    use(((a, b) := Tuple(1, 2))[0])
    use(a)
    use(b)


def walrus_tuple_target_hoisted():
    var a: Int
    var b: Int
    use(((a, b) := Tuple(1, 2))[0])
    use(a)
    use(b)


# Each target of a mixed chain answers for itself, so the plain one keeps the
# name message and its fixit. The other order is not a shape to pin:
# `a := b = 1` does not parse.
def walrus_in_chain():
    # expected-warning @+2 {{implicit declaration of 'c' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+1 {{implicit declaration of 'd' is deprecated; add 'var' before the name}}
    d = c := 5
    use(c)
    use(d)


# An assignment whose name resolves only to something immutable outside this
# scope declares a local instead of assigning it.
def shadow_module_level():
    # expected-warning @+1 {{implicit declaration of 'one' is deprecated; add 'var' before the name}}
    one = 2
    use(one)


# The shadowed declaration need not be at module level: an enclosing function's
# immutable argument qualifies too.
def shadow_enclosing_argument(p: Int):
    def inner():
        # expected-warning @+1 {{implicit declaration of 'p' is deprecated; add 'var' before the name}}
        p = 2
        use(p)

    inner()


def shadow_in_tuple_target():
    # expected-warning @+2 {{implicit declaration of 'one' is deprecated; add 'var' before the assignment target}}
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; add 'var' before the assignment target}}
    one, x = Tuple(1, 2)
    use(one)
    use(x)


# ===----------------------------------------------------------------------=== #
# Sites inside a nested block, where 'var' in place would be scoped to the block
# instead of to the function.
# ===----------------------------------------------------------------------=== #


def nested_if(c: Bool):
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    if c:
        x = 1
    use(1)


def nested_loop(items: List[Int]):
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    for i in items:
        x = i
    use(1)


def nested_with(cm: ExampleCM):
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    with cm as v:
        x = v
    use(1)


# The declaration is function-scoped wherever it is written, so the second arm
# assigns the same variable and the use after the statement sees it.
def nested_both_arms(c: Bool):
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    if c:
        x = 1
    else:
        x = 2
    use(x)


# The hoisted form that message asks for.
def nested_hoisted(c: Bool):
    var x: Int
    if c:
        x = 1
    else:
        x = 2
    use(x)


# Each name of a nested tuple target hoists on its own, so the hoist message
# replaces the target message.
def nested_tuple_target(c: Bool):
    # expected-warning @+3 {{implicit declaration of 'a' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+2 {{implicit declaration of 'b' is deprecated; declare it with 'var' in the function body}}
    if c:
        a, b = Tuple(1, 2)
    use(1)


def nested_shadow(c: Bool):
    # expected-warning @+2 {{implicit declaration of 'one' is deprecated; declare it with 'var' in the function body}}
    if c:
        one = 2
    use(1)


def nested_while(c: Bool):
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    while c:
        x = 1
    use(1)


def nested_try() raises:
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    try:
        x = one()
    # expected-warning @+2 {{implicit declaration of 'y' is deprecated; declare it with 'var' in the function body}}
    except err:
        y = 1
    use(1)


def nested_comptime_if():
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    comptime if True:
        x = 1
    use(1)


def nested_for_else(items: List[Int]):
    for i in items:
        use(i)
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    else:
        x = 1
    use(1)


# A nested 'def' has its own function body, so its own top level is not nested.
def nested_def_body():
    def inner():
        # expected-warning @+1 {{implicit declaration of 'x' is deprecated; add 'var' before the name}}
        x = 1
        use(x)

    inner()


# ===----------------------------------------------------------------------=== #
# A short-circuit operand, a conditional-expression arm and a comprehension body
# are reachable only by a walrus, and each also gets a block of its own, so both
# rules point the same way there.
# ===----------------------------------------------------------------------=== #


def and_operand(c: Bool) -> Bool:
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    return c and (x := truthy())


def or_operand(c: Bool) -> Bool:
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    return c or (x := truthy())


def conditional_arms(c: Bool):
    # expected-warning @+2 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    # expected-warning @+1 {{implicit declaration of 'y' is deprecated; declare it with 'var' in the function body}}
    var v = (x := 1) if c else (y := 2)
    use(v)


def comprehension_walrus(items: List[Int]):
    # expected-warning @+1 {{implicit declaration of 'x' is deprecated; declare it with 'var' in the function body}}
    var doubled = [(x := i) for i in items]
    use(doubled[0])


# The hoisted form each of those messages asks for.
def and_operand_hoisted(c: Bool) -> Bool:
    var x: Bool
    return c and (x := truthy())


def conditional_arms_hoisted(c: Bool):
    var x: Int
    var y: Int
    var v = (x := 1) if c else (y := 2)
    use(v)


def comprehension_walrus_hoisted(items: List[Int]):
    var x: Int
    var doubled = [(x := i) for i in items]
    use(doubled[0])


# ===----------------------------------------------------------------------=== #
# Forms that already spell out their binding, and must stay silent.
# ===----------------------------------------------------------------------=== #


def explicit_var():
    var x = 1
    use(x)


def for_target(items: List[Int]):
    for i in items:
        use(i)


def for_var_target(items: List[Int]):
    for var i in items:
        use(i)


def with_target(cm: ExampleCM):
    with cm as x:
        use(x)


def except_target() raises:
    try:
        use(one())
    except err:
        pass


def comprehension_target(items: List[Int]):
    var doubled = [i * 2 for i in items]
    use(doubled[0])


def discard():
    _ = one()
