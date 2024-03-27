# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s

alias Int = __mlir_type.index


fn use(lhs: Int, rhs: Int) -> Int:
    return rhs


# COM: Verify that the Closure Impl defined in `main` copies the captures x, y on the heap in the init and frees in the del.

# CHECK: kgen.func @"{{.*}}::`_CI_{{.*}}::__del__{{.*}}"
# CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>, index) memoryOnly>> owned_in_mem) {
# CHECK:      [[CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep %arg0[0]
# CHECK-NEXT: [[CAPTURE_FIELD_PTR:%.*]] = pop.load [[CAPTURE_FIELD_ADD]]
# CHECK-NEXT: pop.aligned_free [[CAPTURE_FIELD_PTR]]

# CHECK:  kgen.func @"{{.*}}::`_CI_{{.*}}::__copyinit__{{.*}}"
# CHECK: pop.aligned_alloc

# CHECK:  kgen.func @"{{.*}}::`_CI_{{.*}}::__init__{{.*}}"
# CHECK-SAME: (%arg0: index, %arg1: index,
# CHECK-SAME: %arg2: !kgen.pointer<struct<(pointer<none>, index) memoryOnly>> init_self,
# CHECK-SAME: %arg3: index borrow) capturing {

# CHECK:         [[MY_CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep %arg2[0]
# CHECK-NEXT:    [[HEAP_CAPTURE_LISTS_PTR:%.*]] = pop.aligned_alloc %idx8, %idx16 : <struct<(index, index)>>
# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_0:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][0] : <struct<(index, index)>>
# CHECK-NEXT:    pop.store %arg0, [[HEAP_CAPTURE_LIST_0]] : !kgen.pointer<index>

# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_1:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][1]
# CHECK-NEXT:    pop.store %arg1, [[HEAP_CAPTURE_LIST_1]]

# CHECK-NEXT:    [[OPAQUE_CAPTURE_LIST:%.*]] = pop.pointer.bitcast [[HEAP_CAPTURE_LISTS_PTR]]
# CHECK-NEXT:    pop.store [[OPAQUE_CAPTURE_LIST]], [[MY_CAPTURE_FIELD_ADD]]
# CHECK-NEXT:    [[NONPARAMETRIC_CAPTURE_ADD:%.*]] = kgen.struct.gep %arg2[1]
# CHECK-NEXT:    pop.store %arg3, [[NONPARAMETRIC_CAPTURE_ADD]] : !kgen.pointer<index>
# CHECK-NEXT:    kgen.return


@no_inline
fn takeClosure(formatter: fn (v: Int) escaping -> Int, value: Int):
    _ = formatter(value)


@no_inline
fn makeEscapingClosure[
    parametricClosure: fn[x: Int] (v: Int) capturing -> Int
](x: Int) -> fn (v: Int) escaping -> Int:
    fn formatter(v: Int) escaping -> Int:
        return parametricClosure[__mlir_attr.`2 : index`](use(x, v))

    return formatter


@export
fn top(a: Int, b: Int):
    var x = use(a, b)
    var y = use(b, a)

    @no_inline
    @__copy_capture(x)
    @parameter
    fn formatter(v: Int) -> Int:
        return use(x, v)

    @no_inline
    @__copy_capture(y)
    @parameter
    fn formatter2[x: Int](v: Int) -> Int:
        return use(y, formatter(v))

    var f = makeEscapingClosure[formatter2](y)
    takeClosure(f, y)
