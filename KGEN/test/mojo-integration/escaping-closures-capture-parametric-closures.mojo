# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s

from sys import argv

# COM: Verify that the Closure Impl defined in `main` copies the captures x, y on the heap in the init and frees in the del.

# CHECK: kgen.func @"{{.*}}::`_CI_{{.*}}::__del__{{.*}}"
# CHECK-SAME: (%arg0: !kgen.pointer<struct<(pointer<none>, index) memoryOnly>> owned_in_mem) -> !kgen.none {
# CHECK:      [[CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep %arg0[0]
# CHECK-NEXT: [[CAPTURE_FIELD_PTR:%.*]] = pop.load [[CAPTURE_FIELD_ADD]]
# CHECK-NEXT: pop.aligned_free [[CAPTURE_FIELD_PTR]]

# CHECK:  kgen.func @"{{.*}}::`_CI_{{.*}}::__copyinit__{{.*}}"
# CHECK: pop.aligned_alloc

# CHECK:  kgen.func @"{{.*}}::`_CI_{{.*}}::__init__{{.*}}"
# CHECK-SAME: (%arg0: !kgen.struct<(index)>, %arg1: !kgen.struct<(index)>,
# CHECK-SAME: %arg2: !kgen.pointer<struct<(pointer<none>, index) memoryOnly>> init_self,
# CHECK-SAME: %arg3: index borrow) capturing -> !kgen.none {
# CHECK-NEXT:    %none = kgen.param.constant: none = <#kgen.none>

# CHECK:         [[MY_CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep %arg2[0]
# CHECK-NEXT:    [[HEAP_CAPTURE_LISTS_PTR:%.*]] = pop.aligned_alloc %idx8, %idx16 : <struct<(struct<(index)>, struct<(index)>)>>
# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_0:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][0] : <struct<(struct<(index)>, struct<(index)>)>>
# CHECK-NEXT:    pop.store %arg0, [[HEAP_CAPTURE_LIST_0]] : !kgen.pointer<struct<(index)>>

# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_1:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][1] : <struct<(struct<(index)>, struct<(index)>)>>
# CHECK-NEXT:    pop.store %arg1, [[HEAP_CAPTURE_LIST_1]] : !kgen.pointer<struct<(index)>>

# CHECK-NEXT:    [[OPAQUE_CAPTURE_LIST:%.*]] = pop.pointer.bitcast [[HEAP_CAPTURE_LISTS_PTR]]
# CHECK-NEXT:    pop.store [[OPAQUE_CAPTURE_LIST]], [[MY_CAPTURE_FIELD_ADD]]
# CHECK-NEXT:    [[NONPARAMETRIC_CAPTURE_ADD:%.*]] = kgen.struct.gep %arg2[1]
# CHECK-NEXT:    pop.store %arg3, [[NONPARAMETRIC_CAPTURE_ADD]] : !kgen.pointer<index>
# CHECK-NEXT:    kgen.return %none : !kgen.none


@no_inline
fn takeClosure(formatter: fn (v: Int) escaping -> Int, value: Int):
    print(formatter(value))


@no_inline
fn makeEscapingClosure[
    parametricClosure: fn[x: Int] (v: Int) capturing -> Int
](x: Int) -> fn (v: Int) escaping -> Int:
    fn formatter(v: Int) escaping -> Int:
        return parametricClosure[2](x + v)

    return formatter


fn main():
    try:
        let x = atol(argv()[1])
        let y = atol(argv()[2])

        @no_inline
        @parameter
        fn formatter(v: Int) -> Int:
            return x + v

        @no_inline
        @parameter
        fn formatter2[x: Int](v: Int) -> Int:
            return y + formatter(v)

        let f = makeEscapingClosure[formatter2](y)
        takeClosure(f, y)
    except e:
        print(e)
