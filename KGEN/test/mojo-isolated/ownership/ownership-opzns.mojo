# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

# Test for CheckLifetimes optimizations.


# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample:
    var x: Int

    fn __init__(inout self):
        self.x = 42
        pass

    fn noop(self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        self.x = existing.x

    fn __copyinit__(inout self, existing: Self):
        self.x = existing.x

    fn __bool__(self) -> Bool:
        return True

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.struct.decl @RegExample
@register_passable
struct RegExample:
    fn __init__(inout self):
        return

    fn __copyinit__(inout self, existing: Self):
        return

    fn noop(self):
        pass

    fn __del__(owned self):
        pass

    fn mutate(inout self):
        pass


# This type is a unique value that cannot be moved without ending lifetime.
# CHECK-LABEL: lit.struct.decl @MemoryUniqueMovable
struct MemoryUniqueMovable:
    var state: MemExample

    fn __init__(inout self):
        self.state = MemExample()

    # CHECK: lit.func @"__moveinit__
    fn __moveinit__(inout self, owned other: Self):
        # Mercilessly steal 'other's state which could be interesting.

        # CHECK-NEXT: %0 = lit.ref.struct.ger %other[state]
        # CHECK-NEXT: lit.ownership.use %0
        # CHECK-NEXT: %1 = lit.ref.struct.ger %self[state]
        # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%1, %0)
        self.state = other.state^

        # CHECK-NEXT: kgen.param.constant: none
        # CHECK-NEXT: lit.ownership.mark_destroyed %other
        # CHECK-NEXT: kgen.return


# This type is copyable/moveable.
# CHECK-LABEL: lit.struct.decl @MemoryMovableCopyable
struct MemoryMovableCopyable:
    var state: MemExample

    fn __init__(inout self):
        self.state = MemExample()

    fn __moveinit__(inout self, owned existing: Self):
        # Mercilessly steal 'existing's state which could be interesting.
        self.state = existing.state^

    fn __copyinit__(inout self, existing: Self):
        self.state = existing.state

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.func @"result_mem1
fn result_mem1(owned a: MemoryUniqueMovable) -> MemoryUniqueMovable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%__result__, %a)
    # CHECK-NEXT: kgen.param.constant: none
    # CHECK-NEXT: kgen.return
    return a^


# CHECK-LABEL: lit.func @"result_mem3
fn result_mem3(owned a: MemoryMovableCopyable) -> MemoryMovableCopyable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%__result__, %a){{.*}}init_self{{.*}} owned_in_mem
    # CHECK-NEXT: kgen.param.constant: none
    # CHECK-NEXT: kgen.return
    return a^


@register_passable
struct RegUniqueMovable:
    fn __init__(inout self):
        return

    fn __del__(owned self):
        pass


@register_passable
struct RegMovableCopyable:
    fn __init__(inout self):
        return

    fn __copyinit__(inout self, existing: Self):
        return

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.func @"result_reg1
fn result_reg1(owned a: RegUniqueMovable) -> RegUniqueMovable:
    # CHECK-NEXT: %a_0 = lit.var.decl "a" arg
    # CHECK-NEXT: lifetime.start %a_0
    # CHECK-NEXT: lit.ref.store %a, %a_0
    # CHECK-NEXT: lit.ownership.use %a_0
    # CHECK-NEXT: [[AVAL:%.*]] = lit.load.consume %a_0
    # CHECK-NEXT: lifetime.end %a_0
    # CHECK-NEXT: kgen.return [[AVAL]]
    return a^


# CHECK-LABEL: lit.func @"result_reg2
fn result_reg2(owned a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: %a_0 = lit.var.decl "a" arg
    # CHECK-NEXT: lifetime.start %a_0
    # CHECK-NEXT: lit.ref.store %a, %a_0
    # CHECK-NEXT: [[A:%.*]] = lit.ref.load %a_0
    # CHECK-NEXT: lifetime.end %a_0
    # CHECK-NEXT: kgen.return [[A]]
    return a


# CHECK-LABEL: lit.func @"result_reg3
fn result_reg3(owned a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: %a_0 = lit.var.decl "a" arg
    # CHECK-NEXT: lifetime.start %a_0
    # CHECK-NEXT: lit.ref.store %a, %a_0
    # CHECK-NEXT: lit.ownership.use %a_0
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a_0
    # CHECK-NEXT: lifetime.end %a_0
    # CHECK-NEXT: kgen.return [[A]]
    return a^


# CHECK-LABEL: lit.func @"result_reg4
fn result_reg4(owned a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: %a_0 = lit.var.decl "a" arg
    # CHECK-NEXT: lifetime.start %a_0
    # CHECK-NEXT: lit.ref.store %a, %a_0

    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: lit.ownership.use %a_0
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a_0
    # CHECK-NEXT: lifetime.end %a_0
    # CHECK-NEXT: lifetime.start %x
    # CHECK-NEXT: lit.ref.store [[A]], %x
    var x = a^

    # CHECK-NEXT: lit.ownership.use %x
    # CHECK-NEXT: [[RES:%.*]] = lit.load.consume %x
    # CHECK-NEXT: lifetime.end %x
    # CHECK-NEXT: kgen.return [[RES]]
    return x^


fn takeOwnedInt(owned x: Int):
    pass


# CHECK-LABEL: lit.func @"passFieldToOwnedInt
fn passFieldToOwnedInt(owned a: MemExample):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %a[x]
    # CHECK-NEXT: %1 = lit.ref.load %0
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)
    # CHECK-NEXT: lit.call {{.*}}takeOwnedInt{{.*}}(%1)
    takeOwnedInt(a.x)

    # CHECK-NEXT: kgen.param.constant: none


# Generic type: Issue #14018
struct MyGenericType[Type: AnyTrivialRegType]:
    var value: Type

    fn __init__(inout self, v: Type):
        self.value = v


fn takeTwo(owned x: RegExample, owned y: RegExample):
    pass


fn takeTwo(owned x: MemExample, owned y: MemExample):
    pass


# Check that copies that are immediately destroyed are elided.
# CHECK-LABEL: lit.func @"optimizeCopyElision
fn optimizeCopyElision():
    # CHECK: %a = lit.var.decl "a"
    # CHECK-NEXT: lifetime.start %a
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%a)
    var a = RegExample()

    # We need one copy of 'a' here, not two + dtor.
    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl
    # CHECK-NEXT: [[A:%.*]] = lit.ref.load %a
    # CHECK-NEXT: lifetime.start [[ANON]]
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[ANON]], [[A]])
    # CHECK-NEXT: [[A:%.*]] = lit.ref.load %a
    # CHECK-NEXT: lifetime.end %a
    # CHECK-NEXT: [[ACOPY:%.*]] = lit.load.consume [[ANON]]
    # CHECK-NEXT: lifetime.end [[ANON]]
    # CHECK-NEXT: lit.call {{.*}}takeTwo{{.*}}([[ACOPY]], [[A]])
    takeTwo(a, a)

    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: lifetime.start %x
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%x)
    var x = MemExample()

    # We need one copy of 'x' here, not two + dtor.

    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: lifetime.start [[ANON]]
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[ANON]], [[IMMREF]])
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: kgen.param.declare
    # CHECK-NEXT: [[PTR:%.*]] = kgen.rebind [[IMMREF]]
    # CHECK-NEXT: lit.call {{.*}}takeTwo{{.*}}([[ANON]], [[PTR]])
    # CHECK-NEXT: lifetime.end %x
    # CHECK-NEXT: lifetime.end [[ANON]]
    takeTwo(x, x)

    # CHECK-NEXT: kgen.param.constant: none


fn consume(owned value: MemExample):
    pass


# CHECK-LABEL: lit.func @"copyElisionArgument
fn copyElisionArgument(owned value: MemExample):
    # CHECK-NEXT: %0 = lit.ref.immut %value
    # CHECK-NEXT: kgen.param.declare
    # CHECK-NEXT: %1 = kgen.rebind %0
    # CHECK-NEXT: call {{.*}}consume{{.*}}(%1)
    # CHECK-NEXT: %none =
    consume(value)


# CHECK-LABEL: lit.func @"optimizeCopyToMove
fn optimizeCopyToMove():
    # All the copy ctors should be eliminated in favor of moves.

    # CHECK: %m1 = lit.var.decl
    # CHECK-NEXT: lifetime.start %m1
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%m1)
    var m1 = MemExample()  # expected-warning {{never mutated}}
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m1.noop()

    # CHECK: %m2 = lit.var.decl
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m1
    # CHECK-NEXT: lifetime.start %m2
    # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%m2, %m1)
    # CHECK-NEXT: lifetime.end %m1
    var m2 = m1  # expected-warning {{never mutated}}
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m2
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m2.noop()

    # CHECK: %m3 = lit.var.decl
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m2
    # CHECK-NEXT: lifetime.start %m3
    # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%m3, %m2)
    # CHECK-NEXT: lifetime.end %m2
    var m3 = m2  # expected-warning {{never mutated}}

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m3
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m3.noop()
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%m3)
    # CHECK-NEXT: lifetime.end %m3

    # All the copyinit's should be removed.

    # CHECK-NEXT: %r1 = lit.var.decl "r1"
    # CHECK-NEXT: lifetime.start %r1
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%r1)
    var r1 = RegExample()
    # CHECK-NEXT: [[R1:%.*]] = lit.ref.load %r1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R1]])
    r1.noop()

    # CHECK-NEXT: %r2 = lit.var.decl "r2"
    # CHECK-NEXT: [[R1:%.*]] = lit.ref.load %r1
    # CHECK-NEXT: lifetime.end %r1
    # CHECK-NEXT: lifetime.start %r2
    # CHECK-NEXT: lit.ref.store [[R1]], %r2
    var r2 = r1
    # CHECK-NEXT: [[R2:%.*]] = lit.ref.load %r2
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R2]])
    r2.noop()

    # CHECK-NEXT: %r3 = lit.var.decl "r3"
    # CHECK-NEXT: [[R2:%.*]] = lit.ref.load %r2
    # CHECK-NEXT: lifetime.end %r2
    # CHECK-NEXT: lifetime.start %r3
    # CHECK-NEXT: lit.ref.store [[R2]], %r3
    var r3 = r2
    # CHECK-NEXT: [[R3:%.*]] = lit.ref.load %r3
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R3]])
    r3.noop()
    # CHECK-NEXT: [[R3:%.*]] = lit.ref.load %r3
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[R3]])
    # CHECK-NEXT: lifetime.end %r3

    # CHECK-NEXT: %v1 = lit.var.decl
    # CHECK-NEXT: lifetime.start %v1
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}(%v1)
    var v1 = RegExample()  # expected-warning {{never mutated}}
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %v1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[TMP]])
    v1.noop()

    # CHECK-NEXT: %v2 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %v1
    # CHECK-NEXT: lifetime.end %v1
    # CHECK-NEXT: lifetime.start %v2
    # CHECK-NEXT: lit.ref.store [[TMP]], %v2
    var v2 = v1  # expected-warning {{never mutated}}
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %v2
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[TMP]])
    v2.noop()

    # CHECK-NEXT: %v3 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %v2
    # CHECK-NEXT: lifetime.end %v2
    # CHECK-NEXT: lifetime.start %v3
    # CHECK-NEXT: lit.ref.store [[TMP]], %v3
    var v3 = v2  # expected-warning {{never mutated}}
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %v3
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[TMP]])
    v3.noop()

    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %v3
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[TMP]])
    # CHECK-NEXT: lifetime.end %v3
    # CHECK-NEXT: kgen.param.constant: none


# This is an integration test for elideCopyDestroyPair
# CHECK-LABEL: lit.func @"optimize_copies
fn optimize_copies() -> MemExample:
    # CHECK: lit.call {{.*}}__init__{{.*}}(%x
    var x = MemExample()

    # Optimized away, so the vardecl is gone, but the lifetime still gets
    # declared.
    # CHECK-NOT: lit.var.decl
    # CHECK: kgen.param.declare *"y`2":
    var y = x
    # CHECK-NOT: lit.var.decl
    # CHECK: kgen.param.declare *"z`3":
    # CHECK-NOT: lit.var.decl
    var z = y
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%__result__,
    return z


# This is not optimized, because there are no destructors for CheckLifetimes
# to insert, so it is a different optimization.


# CHECK-LABEL: lit.func @"optimize_transfers
# Issue #34138
fn optimize_transfers() -> MemExample:
    # CHECK: lit.call {{.*}}__init__{{.*}}(%x
    var x = MemExample()

    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%y, %x)
    var y = x^
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%z, %y)
    var z = y^
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%__result__, %z)
    return z^
