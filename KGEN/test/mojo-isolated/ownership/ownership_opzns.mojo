# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

# Test for CheckLifetimes optimizations.


# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample:
    var x: Int

    fn __init__(out self):
        self.x = 42
        pass

    fn noop(self):
        pass

    fn __moveinit__(out self, owned existing: Self):
        self.x = existing.x

    fn __copyinit__(out self, existing: Self):
        self.x = existing.x

    fn __bool__(self) -> Bool:
        return True

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.struct.decl @RegExample
@register_passable
struct RegExample:
    fn __init__(out self):
        return

    fn __copyinit__(out self, existing: Self):
        return

    fn noop(self):
        pass

    fn __del__(owned self):
        pass

    fn mutate(mut self):
        pass


# This type is a unique value that cannot be moved without ending lifetime.
# CHECK-LABEL: lit.struct.decl @MemoryUniqueMovable
struct MemoryUniqueMovable:
    var state: MemExample

    fn __init__(out self):
        self.state = MemExample()

    # CHECK: lit.fn @"__moveinit__
    fn __moveinit__(out self, owned other: Self):
        # Mercilessly steal 'other's state which could be interesting.

        # CHECK-NEXT: %0 = lit.ref.struct.ger %self[state]
        # CHECK-NEXT: %1 = lit.ref.struct.ger %other[state]
        # CHECK-NEXT: lit.ownership.use %1
        # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%1, %0)
        self.state = other.state^

        # CHECK-NEXT: kgen.param.constant: none
        # CHECK-NEXT: lit.ownership.mark_destroyed %other
        # CHECK-NEXT: kgen.return


# This type is copyable/moveable.
# CHECK-LABEL: lit.struct.decl @MemoryMovableCopyable
struct MemoryMovableCopyable:
    var state: MemExample

    fn __init__(out self):
        self.state = MemExample()

    fn __moveinit__(out self, owned existing: Self):
        # Mercilessly steal 'existing's state which could be interesting.
        self.state = existing.state^

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.fn @"result_mem1
fn result_mem1(owned a: MemoryUniqueMovable) -> MemoryUniqueMovable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%a, %__result__)
    # CHECK-NEXT: kgen.param.constant: none
    # CHECK-NEXT: kgen.return
    return a^


# CHECK-LABEL: lit.fn @"result_mem3
fn result_mem3(owned a: MemoryMovableCopyable) -> MemoryMovableCopyable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: lit.call {{.*}}__moveinit__{{.*}}(%a, %__result__){{.*}} owned_in_mem{{.*}}byref_result
    # CHECK-NEXT: kgen.param.constant: none
    # CHECK-NEXT: kgen.return
    return a^

# CHECK-LABEL: lit.fn @"self_copy
fn self_copy(mut x: MemoryMovableCopyable):
    # Mojo introduces a temporary to avoid exclusivity error.
    # CHECK: %__call_result_tmp__ = lit.var.decl
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%x, %__call_result_tmp__)
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%__call_result_tmp__, %x)
    x = x


@register_passable
struct RegUniqueMovable:
    fn __init__(out self):
        return

    fn __del__(owned self):
        pass


@register_passable
struct RegMovableCopyable:
    fn __init__(out self):
        return

    fn __copyinit__(out self, existing: Self):
        return

    fn __del__(owned self):
        pass


# CHECK-LABEL: lit.fn @"result_reg1
fn result_reg1(owned a: RegUniqueMovable) -> RegUniqueMovable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: [[AVAL:%.*]] = lit.load.consume %a
    # CHECK-NEXT: kgen.return [[AVAL]]
    return a^


# CHECK-LABEL: lit.fn @"result_reg2
fn result_reg2(owned a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a
    # CHECK-NEXT: kgen.return [[A]]
    return a


# CHECK-LABEL: lit.fn @"result_reg3
fn result_reg3(owned a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a
    # CHECK-NEXT: kgen.return [[A]]
    return a^


# CHECK-LABEL: lit.fn @"result_reg4
fn result_reg4(owned a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a
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


# CHECK-LABEL: lit.fn @"passFieldToOwnedInt
fn passFieldToOwnedInt(owned a: MemExample):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %a[x]
    # CHECK-NEXT: %1 = lit.ref.load %0
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%a)
    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous*
    # CHECK-NEXT: lit.var.lifetime.start [[ANON]]
    # CHECK-NEXT: lit.ref.store %1, [[ANON]]
    # CHECK-NEXT: lit.call {{.*}}takeOwnedInt{{.*}}([[ANON]])
    # CHECK-NEXT: lit.var.lifetime.end [[ANON]]
    takeOwnedInt(a.x)

    # CHECK-NEXT: kgen.param.constant: none


# Generic type: Issue #14018
struct MyGenericType[Type: AnyTrivialRegType]:
    var value: Type

    @implicit
    fn __init__(out self, v: Type):
        self.value = v


fn takeTwo(owned x: RegExample, owned y: RegExample):
    pass


fn takeTwo(owned x: MemExample, owned y: MemExample):
    pass


# Check that copies that are immediately destroyed are elided.
# CHECK-LABEL: lit.fn @"optimizeCopyElision
fn optimizeCopyElision():
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK: %a = lit.var.decl "a"
    # CHECK-NEXT: lifetime.start %a
    # CHECK-NEXT: lit.ref.store [[TMP]], %a
    var a = RegExample()

    # We need one copy of 'a' here, not two + dtor.
    # CHECK-NEXT: [[A:%.*]] = lit.ref.immut %a
    # CHECK-NEXT: [[COPY1:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[A]])
    # CHECK-NEXT: [[COPY2:%.*]] = lit.load.consume %a
    # CHECK-NEXT: lifetime.end %a

    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl
    # CHECK-NEXT: lifetime.start [[ANON]]
    # CHECK-NEXT: lit.ref.store [[COPY1]], [[ANON]]

    # CHECK-NEXT: [[ANON2:%.*]] = lit.var.decl
    # CHECK-NEXT: lifetime.start [[ANON2]]
    # CHECK-NEXT: lit.ref.store [[COPY2]], [[ANON2]]

    # CHECK-NEXT: lit.call {{.*}}takeTwo{{.*}}([[ANON]], [[ANON2]])
    takeTwo(a, a)
    # CHECK-NEXT: lifetime.end [[ANON2]]
    # CHECK-NEXT: lifetime.end [[ANON]]

    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: lifetime.start %x
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%x)
    var x = MemExample()

    # We need one copy of 'x' here, not two + dtor.

    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: lifetime.start [[ANON]]
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], [[ANON]])
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: kgen.param.declare
    # CHECK-NEXT: [[PTR:%.*]] = kgen.rebind %x
    # CHECK-NEXT: lit.call {{.*}}takeTwo{{.*}}([[ANON]], [[PTR]])
    # CHECK-NEXT: lifetime.end %x
    # CHECK-NEXT: lifetime.end [[ANON]]
    takeTwo(x, x)

    # CHECK-NEXT: kgen.param.constant: none


fn consume(owned value: MemExample):
    pass


# CHECK-LABEL: lit.fn @"copyElisionArgument
fn copyElisionArgument(owned value: MemExample):
    # CHECK-NEXT: %0 = lit.ref.immut %value
    # CHECK-NEXT: kgen.param.declare
    # CHECK-NEXT: %1 = kgen.rebind %value
    # CHECK-NEXT: call {{.*}}consume{{.*}}(%1)
    # CHECK-NEXT: %none =
    consume(value)


# CHECK-LABEL: lit.fn @"optimizeCopyToMove
fn optimizeCopyToMove():
    # All the copy ctors should be eliminated in favor of moves.

    # CHECK: %m1 = lit.var.decl
    # CHECK-NEXT: lifetime.start %m1
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%m1)
    var m1 = MemExample()
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m1.noop()

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m1
    # CHECK-NEXT: kgen.param.declare *"m2`
    # CHECK-NEXT: [[M2:%.*]] = kgen.rebind %m1
    var m2 = m1
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[M2]]
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m2.noop()

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[M2]]
    # CHECK-NEXT: kgen.param.declare *"m3`
    # CHECK-NEXT: [[M3:%.*]] = kgen.rebind [[M2]]
    var m3 = m2

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[M3]]
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m3.noop()
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}([[M3]])
    # CHECK-NEXT: lit.var.lifetime.end %m1

    # All the copyinit's should be removed.

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: %r1 = lit.var.decl "r1"
    # CHECK-NEXT: lifetime.start %r1
    # CHECK-NEXT: lit.ref.store [[TMP]], %r1
    var r1 = RegExample()
    # CHECK-NEXT: [[R1:%.*]] = lit.ref.immut %r1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R1]])
    r1.noop()

    # CHECK-NEXT: %r2 = lit.var.decl "r2"
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %r1
    # CHECK-NEXT: lit.var.lifetime.end %r1
    # CHECK-NEXT: lit.var.lifetime.start %r2
    # CHECK-NEXT: lit.ref.store [[TMP]], %r2
    var r2 = r1
    # CHECK-NEXT: [[R2I:%.*]] = lit.ref.immut %r2
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R2I]])
    r2.noop()

    # CHECK-NEXT: %r3 = lit.var.decl "r3"
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %r2
    # CHECK-NEXT: lit.var.lifetime.end %r2
    # CHECK-NEXT: lit.var.lifetime.start %r3
    # CHECK-NEXT: lit.ref.store [[TMP]], %r3
    var r3 = r2
    # CHECK-NEXT: [[R3I:%.*]] = lit.ref.immut %r3
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R3I]])
    r3.noop()
    # CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%r3)
    # CHECK-NEXT: lifetime.end %r3


# This is an integration test for elideCopyDestroyPair
# CHECK-LABEL: lit.fn @"optimize_copies
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
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}({{.*}}, %__result__)
    return z


# This is not optimized, because there are no destructors for CheckLifetimes
# to insert, so it is a different optimization.


# CHECK-LABEL: lit.fn @"optimize_transfers
# Issue #34138
fn optimize_transfers() -> MemExample:
    # CHECK: lit.call {{.*}}__init__{{.*}}(%x
    var x = MemExample()

    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%x, %y)
    var y = x^
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%y, %z)
    var z = y^
    # CHECK: lit.call {{.*}}__moveinit__{{.*}}(%z, %__result__)
    return z^
