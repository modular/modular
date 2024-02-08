# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s


struct WeirdArray:
    fn __getitem__(self, x: Int) -> Int:
        return x

    fn __getitem__(self, x: Int, y: Int) -> Int:
        return x

    fn __getitem__(self, x: Int, y: Int, z: Int) -> Int:
        return x

    fn __getitem__(self, x: Float, *ints: Int) -> Int:
        return `1`

    fn __setitem__(self, x: Int, y: Int, value: Int):
        pass

    fn __getitem__(self, s: Slice) -> Int:
        return `2`


# CHECK-LABEL: lit.func @"test_getitem
fn test_getitem(a: WeirdArray, idx: Int, f: Float):
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
    _ = a[idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx)
    _ = a[idx, idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx, %idx)
    _ = a[idx, idx, idx]

    # CHECK: [[VARIADIC:%.*]] = pop.variadic.splat 4, %idx
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %f, [[VARIADIC]])
    _ = a[f, idx, idx, idx, idx]


fn test_getitem_kw(a: WeirdArray, idx: Int, idx2: Int, idx3: Int):
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
    _ = a[x=idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2)
    _ = a[y=idx2, x=idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
    _ = a[z=idx3, x=idx, y=idx2]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
    _ = a[idx, z=idx3, y=idx2]


# CHECK-LABEL: lit.func @"test_setitem
fn test_setitem[x: Int](a: WeirdArray, idx: Int):
    # CHECK: %[[X:.*]] = kgen.param.constant = <x>
    # CHECK: lit.call {{.*}}__setitem__{{.*}}(%a, %idx, %idx, %[[X]])
    a[idx, idx] = x


# CHECK-LABEL: lit.func @"test_getitem_slice
fn test_getitem_slice(a: WeirdArray, i: Int, j: Int, k: Int):
    # CHECK: %[[SLICE:.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}"<:type none, :type none, :type none>
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %[[SLICE]])
    _ = a[:]

    # CHECK: %[[SLICE:.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}"<:type none, :type none, :type none>
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %[[SLICE]])
    _ = a[::]

    # CHECK: %[[NONE:.*]] = kgen.param.constant: none
    # CHECK: %[[SLICE:.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}"<:type index, :type index, :type none>(%i, %j, %[[NONE]])
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %[[SLICE]])
    _ = a[i:j]

    # CHECK: %[[NONE:.*]] = kgen.param.constant: none
    # CHECK: %[[SLICE:.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}"<:type none, :type index, :type index>(%[[NONE]], %i, %j)
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %[[SLICE]])
    _ = a[:i:j]

    # CHECK: %[[SLICE:.*]] = lit.call {{.*}}@Slice::@"__init__{{.*}}"<:type index, :type index, :type index>(%i, %j, %k)
    # CHECK-NEXT: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %[[SLICE]])
    _ = a[i:j:k]


struct IndexArray:
    fn __getitem__(inout self, x: Int) -> Int:
        pass

    fn __setitem__(inout self, x: Int, value: Int):
        pass


struct IndexArrayArray:
    fn __getitem__(inout self, x: Int) -> IndexArray:
        pass

    fn __setitem__(inout self, x: Int, value: IndexArray):
        pass


fn takes_inout_int(inout a: Int):
    pass


# CHECK-LABEL: lit.func @"test_writebacks
fn test_writebacks[
    x: Int, y: Int
](inout a: IndexArray, inout b: IndexArrayArray):
    # CHECK: %[[LT:.*]] = lit.varlet.decl "anonymous*" synth
    # CHECK-NEXT: %[[V0:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V1:.*]] = lit.call {{.*}}__getitem__{{.*}}(%a, %[[V0]])
    # CHECK-NEXT: lit.ref.store %[[V1]], %[[LT]]
    # CHECK-NEXT: %[[V2:.*]] = lit.call {{.*}}takes_inout_int{{.*}}(%[[LT]])
    # CHECK-NEXT: %[[V3:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V4:.*]] = lit.ref.load %[[LT]]
    # CHECK-NEXT: %[[V5:.*]] = lit.call {{.*}}__setitem__{{.*}}(%a, %[[V3]], %[[V4]])
    takes_inout_int(a[x])

    # CHECK: %[[LT1:.*]] = lit.varlet.decl
    # CHECK: %[[LT2:.*]] = lit.varlet.decl {{.*}}!IndexArray
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V4:.*]] = {{.*}}__getitem__{{.*}}(%[[LT2]], %b, %[[C1]])
    # CHECK-NEXT: %[[C2:.*]] = kgen.param.constant = <y>
    # CHECK-NEXT: %[[V5:.*]] = lit.call {{.*}}__getitem__{{.*}}(%[[LT2]], %[[C2]])
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %[[LT2]]
    # CHECK-NEXT: %[[V6:.*]] = lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], [[IMMREF]])
    # CHECK-NEXT: lit.ref.store %[[V5]], %[[LT1]]
    # CHECK-NEXT: %[[V7:.*]] = lit.call {{.*}}takes_inout_int{{.*}}(%[[LT1]])
    # CHECK-NEXT: %[[LT3:.*]] = lit.varlet.decl
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V8:.*]] = lit.call {{.*}}__getitem__{{.*}}(%[[LT3]], %b, %[[C1]])
    # CHECK-NEXT: %[[C2:.*]] = kgen.param.constant = <y>
    # CHECK-NEXT: %[[V9:.*]] = lit.ref.load %[[LT1]]
    # CHECK-NEXT: %[[V10:.*]] = lit.call {{.*}}__setitem__{{.*}}(%[[LT3]], %[[C2]], %[[V9]])
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %[[LT3]]
    # CHECK-NEXT: %[[V11:.*]] = lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], [[IMMREF]])
    takes_inout_int(b[x][y])


@register_passable
struct RegWeirdArray:
    fn __getitem__(self, idx: Int) -> Int:
        return idx

    fn __setitem__(self, idx: Int, value: Int):
        pass


# CHECK-LABEL: lit.func @"test_dlvalue_to_pvalue
fn test_dlvalue_to_pvalue[arr: RegWeirdArray, y: Int]():
    # CHECK-NEXT: lit.alias.decl *"x{{.*}}" = <apply({{.*}}@RegWeirdArray::@"__getitem__{{.*}}", arr, y)>
    alias x = arr[y]
