# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# A lambda desugars to a synthetic anonymous closure constructed at emit time.
# These tests check the constructed IR; lambda.mojo (mojo-integration) checks
# that it also executes.

# COM: Non-capturing lambda: empty capture list, explicit return type.

# The synthetic closure is named `lambda_<n>, scope-qualified under its
# enclosing function, and marked synthetic. Its storage carries no captures.
# CHECK-DAG: lit.struct.decl @"withNoCapture()::`lambda_0::__storage"({{.*}}) register_passable_trivial attributes {synthetic}
# The lambda expression instantiates that scope-qualified storage.
# CHECK-DAG: lit.call @{{.*}}::@"withNoCapture()::`lambda_0::__storage"::@"__init__

# The body (x + 1) is lifted into the closure's __call__, also synthetic.
# CHECK: lit.fn @"`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"[{{.*}}](%{{.*}}: {{.*}} read_mem, |, %x: {{.*}}) capturing -> {{.*}} attributes {{{.*}}sourceName = "`lambda_0"{{.*}}synthetic}
# CHECK: kgen.param.constant{{.*}}<{{{.*}}1}>
# CHECK: lit.call tail @{{.*}}::@"__add__{{.*}}"{{.*}}(%x, %{{.*}})
# CHECK: lit.return


def withNoCapture():
    var f = lambda (x: Int) {} -> Int: x + 1


# // -----

# COM: Default-all capture by mut: `{mut}` captures every used outer var by mut.

# The captured `z` becomes a field of the closure storage struct.
# CHECK-DAG: lit.struct.decl @"withCapturingMut()::`lambda_0::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, mut {{.*}}>

# The lambda instantiates the scope-qualified storage, capturing `z`.
# CHECK: lit.call @{{.*}}::@"withCapturingMut()::`lambda_0::__storage"::@"__init__

# The body (x + z) loads the captured `z` out of storage before adding.
# CHECK: lit.fn @"`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"{{.*}} capturing -> {{.*}} attributes {{{.*}}sourceName = "`lambda_0"{{.*}}synthetic}
# CHECK: lit.ref.struct.ger %{{.*}}[z]
# CHECK: lit.call tail @{{.*}}::@"__add__{{.*}}"{{.*}}(%x, %{{.*}})


def withCapturingMut():
    var z = 3
    var f = lambda (x: Int) {mut} -> Int: x + z


# // -----

# COM: Named capture by imm (a bare `{z}` is equivalent): captured by immutable ref.

# CHECK: lit.struct.decl @"withCapturingRead()::`lambda_0::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {synthetic}
# CHECK: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>


def withCapturingRead():
    var z = 3
    var f = lambda (x: Int) {imm z} -> Int: x + z


# // -----

# COM: Named capture by var: captured by value (owned), so the storage field is the
# COM: value itself, not a reference.

# CHECK: lit.struct.decl @"withCapturingVar()::`lambda_0::__storage"({{.*}}) register_passable_trivial attributes {synthetic}
# CHECK: lit.struct.field z : !Int{{[0-9]*}}


def withCapturingVar():
    var z = 3
    var f = lambda (x: Int) {var z} -> Int: x + z


# // -----

# COM: Multiple named captures with mixed conventions: z by imm, w by mut.

# CHECK: lit.struct.decl @"withCapturingMixed()::`lambda_0::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK-DAG: lit.struct.field w : !lit.ref<{{.*}}, mut {{.*}}>
# The lambda instantiates the scope-qualified storage, capturing `z` and `w`.
# CHECK: lit.call @{{.*}}::@"withCapturingMixed()::`lambda_0::__storage"::@"__init__


def withCapturingMixed():
    var z = 3
    var w = 4
    var f = lambda (x: Int) {imm z, mut w} -> Int: x + z + w


# // -----

# CHECK: lit.struct.decl @"withCapturingOverride()::`lambda_0::__storage"<{{.*}}>({{.*}}) attributes {synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, mut {{.*}}>
# CHECK-DAG: lit.struct.field w : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK: lit.call @{{.*}}List{{.*}}append


def withCapturingOverride():
    var z = List[Int]()
    var w = 4
    var f = lambda (x: Int) {imm, mut z} -> None: z.append(x + w)


# // -----

# COM: Parameter list: `[N: Int]` becomes a closure parameter (mangled into the symbol
# COM: as `[::SIMD[::DType(int), ::SIMDLength(1)]]` and declared as `<N: ...>`).

# CHECK: lit.struct.decl @"withParameter()::`lambda_0::__storage"
# CHECK: lit.fn @"`lambda_0[::SIMD[::DType(int), ::SIMDLength(1)]](::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"<N: !Int{{[0-9]*}}>{{.*}} capturing -> {{.*}}


def withParameter():
    var f = lambda [N: Int](x: Int) {} -> Int: x + N


# // -----

# COM: Effects: `raises` (after the argument list, before the capture list) makes the
# COM: lifted __call__ throwing, with the throws ABI (byref error + bool return).

# CHECK: lit.struct.decl @"withEffect()::`lambda_0::__storage"
# CHECK: lit.fn @"`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"{{.*}}byref_error{{.*}} throws|capturing -> {{.*}}


def withEffect():
    var f = lambda (x: Int) raises {} -> Int: x + 1


# // -----

# CHECK: lit.struct.decl @"withReadArg()::`lambda_0::__storage"
# CHECK: lit.fn @"`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"{{.*}}, %x: !Int{{[0-9]*}}) capturing -> {{.*}}


def withReadArg():
    var f = lambda (imm x: Int) {} -> Int: x + 1


# // -----

# CHECK: lit.struct.decl @"withMutArg()::`lambda_0::__storage"
# CHECK: , %x: !lit.ref<{{.*}}> mut) capturing -> {{.*}}


def withMutArg():
    var f = lambda (mut x: Int) {} -> Int: x


# // -----

# CHECK: lit.struct.decl @"withVarArg()::`lambda_0::__storage"
# CHECK: , %x: !lit.ref<{{.*}}> owned_in_mem) capturing -> {{.*}}


def withVarArg():
    var f = lambda (var x: Int) {} -> Int: x + 1


# // -----

# CHECK: lit.struct.decl @"withRefArg()::`lambda_0::__storage"
# CHECK: , %x: !lit.ref<{{.*}}> ref) capturing -> {{.*}}


def withRefArg():
    var f = lambda (ref x: Int) {} -> Int: x + 1


# // -----

# COM: Variadic arguments: `*args` is a positional pack; `**kwargs` packs into
# COM: an `OwnedKwargsDict` and is splat-forwarded through the closure wrapper.

# CHECK-DAG: lit.struct.decl @"withVariadics()::`lambda_0::__storage"
# CHECK-DAG: lit.fn @"{{.*}}`lambda_0[{{.*}}](::SIMD[::DType(int), ::SIMDLength(1)]*){{.*}}"
# CHECK-DAG: lit.struct.decl @"withVariadics()::`lambda_1::__storage"
# CHECK-DAG: lit.fn @"{{.*}}`lambda_1(kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**){{.*}}"
# CHECK-DAG: lit.struct.decl @"withVariadics()::`lambda_2::__storage"
# CHECK-DAG: lit.fn @"{{.*}}`lambda_2[{{.*}}](::SIMD[::DType(int), ::SIMDLength(1)]*,kwargs:::SIMD[::DType(int), ::SIMDLength(1)]**){{.*}}"


def withVariadics():
    var v = lambda (*args: Int) {} -> Int: 0
    var kw = lambda (var **kwargs: Int) {} -> Int: 0
    var both = lambda (*args: Int, var **kwargs: Int) {} -> Int: 0


# // -----

# COM: Everything at once: parameter + argument convention (`var`) + effects + capture +
# COM: return type. The symbol mangles the parameter (`[::SIMD[::DType(int), ::SIMDLength(1)]]`) and owned arg (`::Int$`),
# COM: declares the parameter `<N: ...>`, captures `z`, and is throwing.

# CHECK: lit.struct.decl @"withEverything()::`lambda_0::__storage"
# CHECK: lit.struct.field z : !lit.ref<{{.*}}, mut {{.*}}>
# CHECK: lit.fn @"`lambda_0[::SIMD[::DType(int), ::SIMDLength(1)]](::SIMD[::DType(int), ::SIMDLength(1)]${{.*}}"<{{.*}}N: !Int{{[0-9]*}}>{{.*}} throws|capturing -> {{.*}}


def withEverything():
    var z = 3
    var f = lambda [N: Int](var x: Int) raises {mut} -> Int: x + N + z


# // -----

# COM: A thin lambda bound to a `comptime` promotes to a free function (no
# COM: storage struct), and the alias folds to that function's literal -- as
# COM: `comptime f = some_def` does. Check the promoted fn's definition and the
# COM: alias's reference to it (`{{.*}}` absorbs the promotion mangling suffix).

# CHECK-DAG: lit.fn @"{{.*}}`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"{{.*}}-> !{{.*}}Int{{[0-9]*}}
# CHECK-DAG: lit.alias.decl {{.*}}func.literal{{.*}}func.symbol<@{{.*}}`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}>


comptime inc = lambda (x: Int) {} -> Int: x + 1


def withComptimeBound() -> Int:
    return inc(1)


# // -----

# COM: Elided return type defaults to `None`, like a `def` with no `->`.

# CHECK-DAG: lit.struct.decl @"withElidedReturn()::`lambda_0::__storage"
# CHECK-DAG: lit.struct.decl @"withElidedReturn()::`lambda_1::__storage"
# CHECK-DAG: lit.struct.field lst : !lit.ref<{{.*}}, mut {{.*}}>
# CHECK: lit.fn @"`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"{{.*}} capturing -> !kgen.none
# CHECK: lit.fn @"`lambda_1(){{.*}}"{{.*}} capturing -> !kgen.none
# CHECK: lit.call @{{.*}}List{{.*}}append


def withElidedReturn():
    var f = lambda (x: Int) {}: None
    var lst = [1]
    var g = lambda {mut}: lst.append(2)


# // -----

# COM: Omitted capture list defaults to `{imm}`: free variables are imm-captured (an
# COM: immutable ref); with no free variables the closure is thin, like an explicit `{}`.
# COM: `multi` imm-captures several free variables at once. (Structs emit in reverse
# COM: order, hence CHECK-DAG.)

# CHECK-DAG: lit.struct.decl @"withOmittedCaptures()::`lambda_0::__storage"({{.*}}) register_passable_trivial attributes {synthetic}
# CHECK-DAG: lit.struct.decl @"withOmittedCaptures()::`lambda_1::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {synthetic}
# CHECK-DAG: lit.struct.decl @"withOmittedCaptures()::`lambda_2::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK-DAG: lit.struct.field w : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK: lit.call @{{.*}}::@"withOmittedCaptures()::`lambda_2::__storage"::@"__init__({{.*}},{{.*}})"{{.*}}("z": {{.*}}, "w": {{.*}} ref, |


def withOmittedCaptures():
    var thin = lambda (x: Int) -> Int: x + 1
    var z = 3
    var w = 4
    var reads = lambda (x: Int) -> Int: x + z
    var multi = lambda (x: Int) -> Int: x + z + w


# // -----

# COM: Everything together, with elision: parameter + owned arg + effects, an omitted
# COM: capture list (`z` imm-captured) and an omitted return type (`None`). The body is a
# COM: `None`-returning call that reads the captured `z`.

# CHECK: lit.struct.decl @"withEverythingAndWithElision()::`lambda_0::__storage"
# CHECK: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK: lit.fn @"`lambda_0[::SIMD[::DType(int), ::SIMDLength(1)]](::SIMD[::DType(int), ::SIMDLength(1)]${{.*}}"<{{.*}}N: !Int{{[0-9]*}}>{{.*}}!lit.ref<none, {{.*}}> byref_result{{.*}} throws|capturing -> {{.*}}


def noop(v: Int):
    pass


def withEverythingAndWithElision():
    var z = 3
    var f = lambda [N: Int](var x: Int) raises: noop(z + N + x)


# // -----

# COM: The comptime fold composes with elision: with the capture list omitted and
# COM: nothing captured, the lambda is thin, so it promotes and the alias folds to
# COM: the promoted function's literal -- exactly like the explicit-`{}` fold
# COM: (cf. withComptimeBound). No `__storage` struct exists for it.

# CHECK-DAG: lit.fn @"{{.*}}`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}"{{.*}}-> !{{.*}}Int{{[0-9]*}}
# CHECK-DAG: lit.alias.decl {{.*}}func.literal{{.*}}func.symbol<@{{.*}}`lambda_0(::SIMD[::DType(int), ::SIMDLength(1)]){{.*}}>
# CHECK-NOT: __storage


comptime inc_elided = lambda (x: Int) -> Int: x + 1


def withComptimeBoundElided() -> Int:
    return inc_elided(1)
