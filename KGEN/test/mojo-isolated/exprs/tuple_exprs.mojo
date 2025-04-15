# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

##===----------------------------------------------------------------------===##
# RValue tests
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"tuples_rv
fn tuples_rv(a: Int, b: FloatDyn):
    # CHECK: [[PACK0:%.*]] = kgen.param.constant: !lit.ref.pack
    # CHECK-SAME: <:variadic<!AnyType> [], imm {}> = <<>>
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl{{.*}}@Tuple<:variadic<!AnyType> []>,
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}({{.*}}, [[TMPVAR]])
    _ = ()

    # CHECK-NEXT: [[AREF:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.ref.store %a, [[AREF]]
    # CHECK-NEXT: [[AIMM:%.*]] = lit.ref.immut [[AREF]] : <!Int, mut [[ALT:.*]]>
    # CHECK-NEXT: [[BREF:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.ref.store %b, [[BREF]]
    # CHECK-NEXT: [[BIMM:%.*]] = lit.ref.immut [[BREF]] : <!FloatDyn, mut [[BLT:.*]]>
    # CHECK-NEXT: [[AREBOUND:%.*]] = kgen.rebind [[AIMM]] : !lit.ref<!Int, muttoimm [[ALT]]> to !lit.ref<!Int, imm {(mutcast mut [[ALT]]), (mutcast mut [[BLT]])}>
    # CHECK-NEXT: [[BREBOUND:%.*]] = kgen.rebind [[BIMM]] : !lit.ref<!FloatDyn, muttoimm [[BLT]]> to !lit.ref<!FloatDyn, imm {(mutcast mut [[ALT]]), (mutcast mut [[BLT]])}>
    # CHECK-NEXT: = lit.ref.pack.create([[AREBOUND]], [[BREBOUND]])
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}({{.*}}, [[TMPVAR]])
    _ = (a, b)

    # CHECK: = lit.ref.pack.create({{%[0-9]+}}, {{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}({{.*}}, [[TMPVAR]])
    _ = a, b

    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}, [[TMPVAR]])
    _ = (a,)

    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: lit.call @{{.*}}@Tuple::@"__init__({{.*}}, [[TMPVAR]])
    _ = (a,)

    # CHECK: %c = lit.var.decl "c"
    # CHECK:  = lit.ref.pack.create({{%[0-9]+}})
    # CHECK: [[TUP2:%.*]] = lit.call @{{.*}}@Tuple::@"__init__({{.*}}({{.*}}, %c)
    var c = a,


##===----------------------------------------------------------------------===##
# LValue tests
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"tuples_lv
fn tuples_lv(i0: Int, f0: FloatDyn):
    var i1 = 1
    var i2 = 2

    # CHECK: %iTup = lit.var.decl "iTup"
    var iTup: (Int, Int)

    # Tuple Rvalue
    # CHECK: [[TUP:%.*]] = lit.call {{.*}}@Tuple::@"__init__{{.*}}({{.*}}, %iTup)
    iTup = (i1, i2)

    # Tuple LValue
    # CHECK: [[IMMTUP:%.*]] = lit.ref.immut %iTup
    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}([[IMMTUP]])
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELT]]
    # CHECK: lit.ref.store [[ELTV]], %i1

    # CHECK: [[IMMTUP:%.*]] = lit.ref.immut %iTup
    # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}([[IMMTUP]])
    # CHECK: [[ELTV:%.*]] = lit.ref.load [[ELT]]
    # CHECK: lit.ref.store [[ELTV]], %i2
    (i1, i2) = iTup

    # Check that the swap idiom is correct, this requires producing a copy of the
    # whole RValue on the right before extracting from it.

    # CHECK:  = lit.ref.pack.create
    # CHECK: [[TMPVAR:%.*]] = lit.var.decl {{.*}}@Tuple
    # CHECK: [[TUPRV:%.*]] = lit.call {{.*}}__init__{{.*}}({{.*}}, [[TMPVAR]])

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

    fn __setitem__(mut self, i: Int, owned value: T):
        self.x = value

    fn __getitem__(self, i: Int) -> T:
        return self.x


# CHECK-LABEL: lit.fn @"swap_container_fields
fn swap_container_fields(mut v: Container[_]):
    v[0], v[1] = v[1], v[0]


##===----------------------------------------------------------------------===##
# Tuple Types
##===----------------------------------------------------------------------===##

# FIXME: Empty tuple `Tuple[]` cannot be spelled.


# CHECK-LABEL: lit.fn @"returnTup0
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> []>
fn returnTup0() -> Tuple:
    # FIXME: Why isn't this a kgen.param.constant for the whole call?
    # CHECK: !lit.ref.pack<:variadic<!AnyType> [], imm {}> = <<>>
    return ()


# CHECK-LABEL: lit.fn @"returnTup0a
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> []>
fn returnTup0a() -> ():
    # FIXME: Why isn't this a kgen.param.constant for the whole call?
    # CHECK: kgen.param.constant: !lit.ref.pack<:variadic<!AnyType> [], imm {}> = <<>>
    # CHECK: lit.call{{.*}}__init__
    return ()


# CHECK-LABEL: lit.fn @"returnTup1
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1]>,
fn returnTup1() -> Tuple[Int]:
    # CHECK: %0 = kgen.param.constant: !Int
    # CHECK:   = lit.ref.pack.create({{.*}}) : !lit.ref.pack<:variadic<!AnyType> [#Int1],
    # CHECK:  = lit.call{{.*}}__init__
    return (Int(4),)


# CHECK-LABEL: lit.fn @"returnTup1
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1]>
fn returnTup1a() -> (Int,):
    return (Int(4),)


fn returnTup1b() -> (Int,):
    return (Int(4),)


# CHECK-LABEL: lit.fn @"returnTup2
# CHECK-SAME:  %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1, #FloatDyn1]>
fn returnTup2() -> Tuple[Int, FloatDyn]:
    # CHECK:  = kgen.param.constant: !Int = <{4}>
    # CHECK:  = kgen.param.constant: !FloatDyn = <{:scalar<f64> "2"}>
    # CHECK: lit.ref.pack.create({{.*}}) : !lit.ref.pack<:variadic<!AnyType> [#Int1, #FloatDyn1]
    return (Int(4), 2.0)


# CHECK-LABEL: lit.fn @"returnTup2a
# CHECK-SAME: %__result__: !lit.ref<{{.*}}@Tuple<:variadic<!AnyType> [#Int1, #FloatDyn1]>,
fn returnTup2a() -> (Int, FloatDyn):
    # CHECK: lit.ref.pack.create({{.*}}) : !lit.ref.pack<:variadic<!AnyType> [#Int1, #FloatDyn1]
    return (Int(4), 2.0)


# CHECK-LABEL: lit.fn @"returnTup2b
fn returnTup2b() -> (Int, FloatDyn):
    return Int(4), 2.0


# CHECK-LABEL: lit.fn @"takesSugarTuple{{.*}}<T: !Copyable>
# CHECK-SAME: @Tuple<:variadic<!AnyType> [#type_value1, #type_value1]>
fn takesSugarTuple[T: Copyable](elements: (T, T)):
    pass


# CHECK-LABEL: lit.fn @"index_homogenous_tuple
fn index_homogenous_tuple[idx: Int]():
    var tup = (1, 2, 3, 4)
    # CHECK: %test1 = lit.var.decl "test1"
    # CHECK-NEXT: [[ELTPTR:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}:!Int {1}{{.*}}(%tup)
    # CHECK-NEXT: [[INTVAL:%.*]] = lit.ref.load [[ELTPTR]]
    # CHECK-NEXT: lit.ref.store [[INTVAL]], %test1
    var test1: Int = tup[1]

    # CHECK: %test2 = lit.var.decl "test2"
    # CHECK-NEXT: [[ELTPTR:%.*]] = lit.call {{.*}}Tuple::@"__getitem__{{.*}}:!Int idx{{.*}}(%tup)
    # CHECK-NEXT: [[INTVAL:%.*]] = lit.ref.load [[ELTPTR]]
    # CHECK-NEXT: lit.ref.store [[INTVAL]], %test2
    var test2: Int = tup[idx]
