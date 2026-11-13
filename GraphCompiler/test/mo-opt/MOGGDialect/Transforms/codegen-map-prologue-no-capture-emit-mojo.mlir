// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// A constant producer fuses into the consumer's prologue, and a constant loop
// has no inputs to become captures. The functor then has no fields and reads
// nothing: the value is a literal in its `__call__`, and the constant buffer is
// never materialized. The consumer is a gather, whose data operand takes a
// prologue.

// CHECK: def {{__[A-Za-z0-9_]+}}(
// The captureless functor takes no parameters and no constructor arguments.
// CHECK: var [[F0:.*]] = [[FN0:InputLoadFunctor_[0-9]+]]()
// CHECK: _MoggInputLoadFusionTensor[{{.*}}]([[F0]],

// CHECK: struct [[FN0]](InputLoadFusion, TrivialRegisterPassable):
// No `var` field between the struct and its `__call__`: nothing is captured.
// CHECK-NOT: var
// CHECK: def __call__[
// CHECK: return rebind[SIMD[dtype, simd_width]](SIMD[DType.int32, simd_width](7))

mo.graph @gather_constant(%indices: !mo.tensor<[2], si32>) -> (!mo.tensor<[2, 3], si32>) {
  %c = mo.constant <#M.dense_array<7> : tensor<si32>> : !mo.tensor<[], si32>
  %data = mo.static.broadcast_to(%c) : !mo.tensor<[], si32> -> !mo.tensor<[3, 3], si32>
  %out = mo.gather(%data, %indices) [axis = 0] : (
    !mo.tensor<[3, 3], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[2, 3], si32>
  mo.output %out : !mo.tensor<[2, 3], si32>
}
