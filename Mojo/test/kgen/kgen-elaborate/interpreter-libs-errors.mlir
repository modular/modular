// RUN: not kgen %s -elaborate -S -o - --interpreter-libs=%interpreter_lib 2>&1 | FileCheck %s

// CHECK-DAG: unable to interpret call to unknown external function: no_such_symbol
kgen.generator @unknown_symbol() -> index {
  %0 = pop.external_call @no_such_symbol() : () -> index
  hlcf.return %0 : index
}

kgen.generator export @call_unknown_symbol() -> index {
  %0 = kgen.param.constant = <apply(:() -> index @unknown_symbol)>
  hlcf.return %0 : index
}

// CHECK-DAG: cannot call 'sum_indirect' at compile time: it must not be variadic, take at most 6 pointers or 64-bit integers, and return a 64-bit integer or nothing
kgen.generator @too_many_args(%arg0: index) -> index {
  %0 = pop.external_call @sum_indirect(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0)
    : (index, index, index, index, index, index, index) -> index
  hlcf.return %0 : index
}

kgen.generator export @call_too_many_args() -> index {
  %0 = kgen.param.constant = <apply(:(index) -> index @too_many_args, 1)>
  hlcf.return %0 : index
}

// COM: The result has 8 bytes of integers but is not one integer.
// CHECK-DAG: cannot call 'sum_indirect' at compile time: it must not be variadic, take at most 6 pointers or 64-bit integers, and return a 64-bit integer or nothing
kgen.generator @wide_result(%arg0: index) -> index {
  %0 = pop.external_call @sum_indirect(%arg0) : (index) -> !kgen.simd<2, si32>
  hlcf.return %arg0 : index
}

kgen.generator export @call_wide_result() -> index {
  %0 = kgen.param.constant = <apply(:(index) -> index @wide_result, 1)>
  hlcf.return %0 : index
}

// CHECK-DAG: cannot call 'sum_indirect' at compile time with an argument that is not a pointer or a 64-bit integer
kgen.generator @narrow_arg(%arg0: index) -> index {
  %0 = index.cmp eq(%arg0, %arg0)
  %1 = pop.external_call @sum_indirect(%0) : (i1) -> index
  hlcf.return %1 : index
}

kgen.generator export @call_narrow_arg() -> index {
  %0 = kgen.param.constant = <apply(:(index) -> index @narrow_arg, 1)>
  hlcf.return %0 : index
}
