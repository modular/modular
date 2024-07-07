# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct WeirdArray:
    fn __getitem__(self, x: int) -> int:
        return x

    fn __getitem__(self, x: int, y: int) -> int:
        return x

    fn __getitem__(self, x: int, y: int, z: int) -> int:
        return x

    fn __getitem__(self, x: float, *ints: int) -> int:
        return `1`

    fn __setitem__(self, x: int, y: int, value: int):
        pass

    fn __getitem__(self, s: Slice) -> int:
        return `2`


# CHECK-LABEL: lit.func @"test_getitem
fn test_getitem(a: WeirdArray, idx: int, f: float):
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
    _ = a[idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx)
    _ = a[idx, idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx, %idx)
    _ = a[idx, idx, idx]

    # CHECK: [[VARIADIC:%.*]] = pop.variadic.splat 4, %idx
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %f, [[VARIADIC]])
    _ = a[f, idx, idx, idx, idx]


fn test_getitem_kw(a: WeirdArray, idx: int, idx2: int, idx3: int):
    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
    _ = a[x=idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2)
    _ = a[y=idx2, x=idx]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
    _ = a[z=idx3, x=idx, y=idx2]

    # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
    _ = a[idx, z=idx3, y=idx2]


# CHECK-LABEL: lit.func @"test_setitem
fn test_setitem[x: int](a: WeirdArray, idx: int):
    # CHECK: %[[X:.*]] = kgen.param.constant = <x>
    # CHECK: lit.call {{.*}}__setitem__{{.*}}(%a, %idx, %idx, %[[X]])
    a[idx, idx] = x


# CHECK-LABEL: lit.func @"test_getitem_slice
fn test_getitem_slice(a: WeirdArray, i: int, j: int, k: int):
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
    fn __getitem__(inout self, x: int) -> int:
        pass

    fn __setitem__(inout self, x: int, value: int):
        pass


struct IndexArrayArray:
    fn __getitem__(inout self, x: int) -> IndexArray:
        pass

    fn __setitem__(inout self, x: int, value: IndexArray):
        pass


fn takes_inout_int(inout a: int):
    pass


# CHECK-LABEL: lit.func @"test_writebacks
fn test_writebacks[
    x: int, y: int
](inout a: IndexArray, inout b: IndexArrayArray):
    # CHECK: %[[LT:.*]] = lit.var.decl "anonymous*" synth
    # CHECK-NEXT: %[[V0:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V1:.*]] = lit.call {{.*}}__getitem__{{.*}}(%a, %[[V0]])
    # CHECK-NEXT: lit.ref.store %[[V1]], %[[LT]]
    # CHECK-NEXT: %[[V2:.*]] = lit.call {{.*}}takes_inout_int{{.*}}(%[[LT]])
    # CHECK-NEXT: %[[V3:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V4:.*]] = lit.ref.load %[[LT]]
    # CHECK-NEXT: %[[V5:.*]] = lit.call {{.*}}__setitem__{{.*}}(%a, %[[V3]], %[[V4]])
    takes_inout_int(a[x])

    # CHECK: %[[LT1:.*]] = lit.var.decl
    # CHECK: %[[LT2:.*]] = lit.var.decl {{.*}}!IndexArray
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V4:.*]] = {{.*}}__getitem__{{.*}}(%b, %[[C1]], %[[LT2]])
    # CHECK-NEXT: %[[C2:.*]] = kgen.param.constant = <y>
    # CHECK-NEXT: %[[V5:.*]] = lit.call {{.*}}__getitem__{{.*}}(%[[LT2]], %[[C2]])
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %[[LT2]]
    # CHECK-NEXT: %[[V6:.*]] = lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], [[IMMREF]])
    # CHECK-NEXT: lit.ref.store %[[V5]], %[[LT1]]
    # CHECK-NEXT: %[[V7:.*]] = lit.call {{.*}}takes_inout_int{{.*}}(%[[LT1]])
    # CHECK-NEXT: %[[LT3:.*]] = lit.var.decl
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: %[[V8:.*]] = lit.call {{.*}}__getitem__{{.*}}(%b, %[[C1]], %[[LT3]])
    # CHECK-NEXT: %[[C2:.*]] = kgen.param.constant = <y>
    # CHECK-NEXT: %[[V9:.*]] = lit.ref.load %[[LT1]]
    # CHECK-NEXT: %[[V10:.*]] = lit.call {{.*}}__setitem__{{.*}}(%[[LT3]], %[[C2]], %[[V9]])
    # CHECK-NEXT: %[[C1:.*]] = kgen.param.constant = <x>
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %[[LT3]]
    # CHECK-NEXT: %[[V11:.*]] = lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], [[IMMREF]])
    takes_inout_int(b[x][y])


@register_passable
struct RegWeirdArray:
    fn __getitem__(self, idx: int) -> int:
        return idx

    fn __setitem__(self, idx: int, value: int):
        pass


# CHECK-LABEL: lit.func @"test_dlvalue_to_pvalue
fn test_dlvalue_to_pvalue[arr: RegWeirdArray, y: int]():
    # CHECK-NEXT: lit.alias.decl *"x{{.*}}" = <apply({{.*}}@RegWeirdArray::@"__getitem__{{.*}}", arr, y)>
    alias x = arr[y]




struct XYZ:
   fn __getattr__[name: StringLiteral](self) -> Int:
      @parameter
      if name == "x":
        return 4
      elif name == "y":
        return 6
      else:
        # Constrained is not supported with stubs library.
        #constrained[name == "z", "can only index with x, y, or z"]()
        return 8
struct ParamIndex:
  fn __getitem__[a: Int, b: Int](self) -> Int: return 42


# CHECK-LABEL: lit.func @"test_param_indexing
fn test_param_indexing(a: XYZ, b: ParamIndex) -> Int:
  # Issue #35662: Support parameter input to getattr
  # CHECK: lit.call {{.*}}__getattr__{{.*}}<:!StringLiteral {:string "x"}>(%a)
  _ = a.x 
  # CHECK: lit.call {{.*}}__getattr__{{.*}}<:!StringLiteral {:string "y"}>(%a)
  _ = a.y
  # CHECK: lit.call {{.*}}__getitem__{{.*}}<:!Int {2}, :!Int {4}>(%b)
  _ = b[2, 4]

# ===----------------------------------------------------------------------=== #
# Keyword arguments in setters

@value
struct VariadicIndexList:
    fn __getitem__(inout self, *indices: Int) -> Int:
        pass

    fn __setitem__(inout self, *indices: Int, val: Int):
        pass

# CHECK-LABEL: lit.func @"testVariadicIndexList
# MOCO-696: Support variadic length keys in __setitem__
fn testVariadicIndexList(inout foo: VariadicIndexList, i: Int, the_value: Int):
    # Getter is straight-forward.
    # CHECK: [[VARIADIC:%.*]] = pop.variadic.splat 2, %i
    # CHECK: lit.call {{.*}}VariadicIndexList::@"__getitem__{{.*}}(%foo, [[VARIADIC]])
    _ = foo[i, i]

    # Setter needs to pass the new value as 'val', not in the variadics.
    # CHECK-NEXT: [[VARIADIC:%.*]] = pop.variadic.splat 4, %i
    # CHECK: lit.call {{.*}}VariadicIndexList::@"__setitem__{{.*}}(%foo, [[VARIADIC]], %the_value)
    foo[i, i, i, i] = the_value

