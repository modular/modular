# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

##===----------------------------------------------------------------------===##
# RValue tests
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"tuples_rv
fn tuples_rv(a: Int, b: FloatDyn):
    # CHECK: [[PACK0:%.*]] = kgen.param.constant: !lit.ref.pack
    # CHECK-SAME: <:variadic<!AnyType> [], imm #lit.lifetime> = <<>>
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}([[TMPVAR]]
    _ = ()

    # CHECK-NEXT: [[APTR:%.*]] = pop.stack_allocation 1 x !Int
    # CHECK-NEXT: lifetime.start([[APTR]])
    # CHECK-NEXT: pop.store %a, [[APTR]] : !kgen.pointer<!Int>
    # CHECK-NEXT: [[AREF:%.*]] = lit.ref.from_pointer [[APTR]] : <!Int, imm #lit.lifetime>
    # CHECK-NEXT: [[BPTR:%.*]] = pop.stack_allocation 1 x !FloatDyn
    # CHECK-NEXT: lifetime.start([[BPTR]])
    # CHECK-NEXT: pop.store %b, [[BPTR]] : !kgen.pointer<!FloatDyn>
    # CHECK-NEXT: [[BREF:%.*]] = lit.ref.from_pointer [[BPTR]]
    # CHECK-NEXT: = lit.ref.pack.create([[AREF]], [[BREF]])
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}([[TMPVAR]]
    _ = (a, b)
    # CHECK-NEXT: ownership.use %a
    # CHECK-NEXT: lifetime.end([[APTR]])
    # CHECK-NEXT: ownership.use %b
    # CHECK-NEXT: lifetime.end([[BPTR]])

    # CHECK: = lit.ref.pack.create({{%[0-9]+}}, {{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}([[TMPVAR]]
    _ = a, b

    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}([[TMPVAR]]
    _ = (a,)

    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}([[TMPVAR]]
    _ = (a,)

    # CHECK: %c = lit.var.decl "c"
    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TUP2:%.*]] = lit.call @{{.*}}@Tuple::@"__init__({{.*}}(%c
    var c = a,


##===----------------------------------------------------------------------===##
# LValue tests
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"tuples_lv
fn tuples_lv(i0: Int, f0: FloatDyn):
    var i1 = 1
    var i2 = 2

    # CHECK: %iTup = lit.var.decl "iTup"
    var iTup: (Int, Int)

    # Tuple Rvalue
    # CHECK: [[TUP:%.*]] = lit.call {{.*}}@Tuple::@"__init__{{.*}}(%iTup,
    iTup = (i1, i2)

    # Tuple LValue
    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}(%iTup)
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELT]]
    # CHECK: lit.ref.store [[ELTV]], %i1

    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}(%iTup)
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELT]]
    # CHECK: lit.ref.store [[ELTV]], %i2
    (i1, i2) = iTup

    # Check that the swap idiom is correct, this requires producing a copy of the
    # whole RValue on the right before extracting from it.

    # CHECK:  = lit.ref.pack.create
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: [[TUPRV:%.*]] = lit.call {{.*}}__init__{{.*}}([[TMPVAR]],

    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}>(
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELT]]
    # CHECK: lit.ref.store [[ELTV]], %i1

    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}>(
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELT]]
    # CHECK: lit.ref.store [[ELTV]], %i2
    (i1, i2) = (i2, i1)

    # CHECK: [[ELT:%.*]] = lit.call {{.*}}__getitem__{{.*}}(%iTup)
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %i1
    # CHECK-NEXT: lit.ref.store [[TMP]], [[ELT]]
    iTup[1] = i1

    var f1: FloatDyn
    # Mixed element types should work.  Don't need check lines though.
    (i1, f1) = (i0, f0)


##===----------------------------------------------------------------------===##
# Memory-only element tests
##===----------------------------------------------------------------------===##


trait CollectionType(Copyable, Movable):
    pass


struct Container[T: CollectionType]:
    var x: T

    fn __setitem__(inout self, i: Int, owned value: T):
        self.x = value

    fn __getitem__(self, i: Int) -> T:
        return self.x


# CHECK-LABEL: lit.func @"swap_container_fields
fn swap_container_fields(inout v: Container[_]):
    v[0], v[1] = v[1], v[0]


##===----------------------------------------------------------------------===##
# Tuple Types
##===----------------------------------------------------------------------===##

# FIXME: Empty tuple `Tuple[]` cannot be spelled.


# CHECK-LABEL: lit.func @"returnTup0
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> []>
fn returnTup0() -> Tuple:
    # FIXME: Why isn't this a kgen.param.constant for the whole call?
    # CHECK: !lit.ref.pack<:variadic<!AnyType> [], imm #lit.lifetime> = <<>>
    return ()


# CHECK-LABEL: lit.func @"returnTup0a
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> []>
fn returnTup0a() -> ():
    # FIXME: Why isn't this a kgen.param.constant for the whole call?
    # CHECK: kgen.param.constant: !lit.ref.pack<:variadic<!AnyType> [], imm #lit.lifetime> = <<>>
    # CHECK: lit.call{{.*}}__init__
    return ()


# CHECK-LABEL: lit.func @"returnTup1
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1]>,
fn returnTup1() -> Tuple[Int]:
    # CHECK: %0 = kgen.param.constant: !Int
    # CHECK:   = lit.ref.pack.create({{.*}}) : !lit.ref.pack<:variadic<!AnyType> [#Int1],
    # CHECK:  = lit.call{{.*}}__init__
    return (Int(4),)


# CHECK-LABEL: lit.func @"returnTup1
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1]>
fn returnTup1a() -> (Int,):
    return (Int(4),)


fn returnTup1b() -> (Int,):
    return (Int(4),)


# CHECK-LABEL: lit.func @"returnTup2
# CHECK-SAME:  %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1, #FloatDyn1]>
fn returnTup2() -> Tuple[Int, FloatDyn]:
    # CHECK:  = kgen.param.constant{{.*}}4
    # CHECK:  = kgen.param.constant{{.*}}:f64 2.0
    # CHECK: lit.ref.pack.create({{.*}}) : !lit.ref.pack<:variadic<!AnyType> [#Int1, #FloatDyn1]
    return (Int(4), 2.0)


# CHECK-LABEL: lit.func @"returnTup2a
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1, #FloatDyn1]>,
fn returnTup2a() -> (Int, FloatDyn):
    # CHECK: lit.ref.pack.create({{.*}}) : !lit.ref.pack<:variadic<!AnyType> [#Int1, #FloatDyn1]
    return (Int(4), 2.0)


# CHECK-LABEL: lit.func @"returnTup2b
fn returnTup2b() -> (Int, FloatDyn):
    return Int(4), 2.0


# CHECK-LABEL: lit.func @"takesSugarTuple{{.*}}<T: !Copyable>
# CHECK-SAME: @Tuple<:variadic<!AnyType> [#type_value1, #type_value1]>
fn takesSugarTuple[T: Copyable](elements: (T, T)):
    pass
