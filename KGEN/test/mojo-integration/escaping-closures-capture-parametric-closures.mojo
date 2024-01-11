# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s

from sys import argv

# COM: Verify that the Closure Impl defined in `main` copies the captures x, y on the heap in the init and frees in the del.

# CHECK: kgen.func @"${{.*}}::`_CI_${{.*}}::__del__{{.*}}"
# CHECK-SAME: (%arg0: !kgen.pointer<struct<(index, pointer<struct<(struct<(index)>, struct<(index)>)>>) memoryOnly>> owned_in_mem) -> !kgen.none {
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: [[CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep %arg0[1] : <struct<(index, pointer<struct<(struct<(index)>, struct<(index)>)>>) memoryOnly>>
# CHECK-NEXT: [[CAPTURE_FIELD_PTR:%.*]] = pop.load [[CAPTURE_FIELD_ADD]] : !kgen.pointer<pointer<struct<(struct<(index)>, struct<(index)>)>>>
# CHECK-NEXT: pop.aligned_free [[CAPTURE_FIELD_PTR]] : <struct<(struct<(index)>, struct<(index)>)>>
# CHECK-NEXT: kgen.return %none : !kgen.none

# CHECK:  kgen.func @"${{.*}}::`_CI_${{.*}}::__init__{{.*}}"
# CHECK-SAME: (%arg0: !kgen.struct<(index)>, %arg1: !kgen.struct<(index)>,
# CHECK-SAME: %arg2: !kgen.pointer<struct<(index, pointer<struct<(struct<(index)>, struct<(index)>)>>) memoryOnly>> init_self,
# CHECK-SAME: %arg3: index borrow) capturing -> !kgen.none {
# CHECK-NEXT:    %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:    %idx8 = index.constant 8
# CHECK-NEXT:    %idx16 = index.constant 16

# CHECK-NEXT:    [[HEAP_CAPTURE_LISTS_PTR:%.*]] = pop.aligned_alloc %idx8, %idx16 : <struct<(struct<(index)>, struct<(index)>)>>
# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_0:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][0] : <struct<(struct<(index)>, struct<(index)>)>>
# CHECK-NEXT:    [[STACK_FIELD_X:%.*]] = kgen.struct.extract %arg0[0] : !kgen.struct<(index)>
# CHECK-NEXT:    [[HEAP_FIELD_ADD_X:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LIST_0]][0] : <struct<(index)>>
# CHECK-NEXT:    pop.store [[STACK_FIELD_X]], [[HEAP_FIELD_ADD_X]] : !kgen.pointer<index>

# CHECK-NEXT:    [[HEAP_CAPTURE_LIST_1:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LISTS_PTR]][1] : <struct<(struct<(index)>, struct<(index)>)>>
# CHECK-NEXT:    [[STACK_FIELD_Y:%.*]] = kgen.struct.extract %arg1[0] : !kgen.struct<(index)>
# CHECK-NEXT:    [[HEAP_FIELD_ADD_Y:%.*]] = kgen.struct.gep [[HEAP_CAPTURE_LIST_1]][0] : <struct<(index)>>
# CHECK-NEXT:    pop.store [[STACK_FIELD_Y]], [[HEAP_FIELD_ADD_Y]] : !kgen.pointer<index>

# CHECK-NEXT:    [[MY_CAPTURE_FIELD_ADD:%.*]] = kgen.struct.gep %arg2[1] : <struct<(index, pointer<struct<(struct<(index)>, struct<(index)>)>>) memoryOnly>>
# CHECK-NEXT:    pop.store [[HEAP_CAPTURE_LISTS_PTR]], [[MY_CAPTURE_FIELD_ADD]] : !kgen.pointer<pointer<struct<(struct<(index)>, struct<(index)>)>>>
# CHECK-NEXT:    [[NONPARAMETRIC_CAPTURE_ADD:%.*]] = kgen.struct.gep %arg2[0] : <struct<(index, pointer<struct<(struct<(index)>, struct<(index)>)>>) memoryOnly>>
# CHECK-NEXT:    pop.store %arg3, [[NONPARAMETRIC_CAPTURE_ADD]] : !kgen.pointer<index>
# CHECK-NEXT:    kgen.return %none : !kgen.none


@no_inline
fn takeClosure(formatter: fn (v: Int) escaping -> Int, value: Int):
    print(formatter(value))


@no_inline
fn makeEscapingClosure[
    parametricClosure: fn (v: Int) capturing -> Int
](x: Int) -> fn (v: Int) escaping -> Int:
    fn formatter(v: Int) escaping -> Int:
        return parametricClosure(x + v)

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
        fn formatter2(v: Int) -> Int:
            return y + formatter(v)

        let f = makeEscapingClosure[formatter2](y)
        takeClosure(f, y)
    except e:
        print(e)
