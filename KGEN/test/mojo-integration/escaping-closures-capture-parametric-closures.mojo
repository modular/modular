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

# CHECK:  kgen.func @"{{.*}}::`_CI_{{.*}}::__copyinit__{{.*}}"
# CHECK: pop.aligned_alloc

# COM: check inlined __init__
# CHECK:  kgen.func @"{{.*}}::makeEscapingClosure
# CHECK:         [[MY_CAPTURE_FIELD_ALLOC:%.*]] = pop.stack_allocation 1 x struct<(pointer<none>, index) memoryOnly>
# CHECK:         lifetime.start([[MY_CAPTURE_FIELD_ALLOC]])
# CHECK-NEXT:    [[MY_CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep [[MY_CAPTURE_FIELD_ALLOC]][0]
# CHECK-NEXT:    [[HEAP_CAPTURE_LISTS_PTR:%.*]] = pop.aligned_alloc %idx8, %idx16 : <struct<(index, index)>>
# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_0:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][0] : <struct<(index, index)>>
# CHECK-NEXT:    pop.store %arg0, [[HEAP_CAPTURE_LIST_0]] : !kgen.pointer<index>

# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_1:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][1]
# CHECK-NEXT:    pop.store %arg1, [[HEAP_CAPTURE_LIST_1]]

# CHECK-NEXT:    [[OPAQUE_CAPTURE_LIST:%.*]] = pop.pointer.bitcast [[HEAP_CAPTURE_LISTS_PTR]]
# CHECK-NEXT:    pop.store [[OPAQUE_CAPTURE_LIST]], [[MY_CAPTURE_FIELD_ADD]]
# CHECK-NEXT:    [[NONPARAMETRIC_CAPTURE_ADD:%.*]] = kgen.struct.gep [[MY_CAPTURE_FIELD_ALLOC]][1]
# CHECK-NEXT:    pop.store %arg2, [[NONPARAMETRIC_CAPTURE_ADD]] : !kgen.pointer<index>

# COM: check inlined __del__
# CHECK: kgen.func @"{{.*}}_dtor_`_CI_{{.*}}"
# CHECK:      [[CAPTURE_FIELD_ARG:%.*]] = pop.pointer.bitcast %arg0
# CHECK-NEXT: [[CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep [[CAPTURE_FIELD_ARG]][0]
# CHECK-NEXT: [[CAPTURE_FIELD_PTR:%.*]] = pop.load [[CAPTURE_FIELD_ADD]]
# CHECK: pop.aligned_free [[CAPTURE_FIELD_PTR]]


@no_inline
fn takeClosure(formatter: fn (v: Int) escaping -> Int, value: Int):
    _ = formatter(value)


@no_inline
fn makeEscapingClosure[
    parametricClosure: fn[x: Int] (v: Int) capturing -> Int
](x: Int) -> fn (v: Int) escaping -> Int:
    fn formatter(v: Int) -> Int:
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
