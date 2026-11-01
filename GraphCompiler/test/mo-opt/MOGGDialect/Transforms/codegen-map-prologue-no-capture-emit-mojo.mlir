// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// A constant producer fuses into the consumer's prologue, and a constant loop
// has no inputs to become captures. The functor then has no fields and reads
// nothing: the value is a literal in its `load`, and the constant buffer is
// never materialized. Contrast the second operand's functor, which captures
// the graph input and loads through it.

// CHECK: def {{__[A-Za-z0-9_]+}}(
// The captureless functor takes no parameters and no constructor arguments.
// CHECK: var [[F0:.*]] = [[FN0:InputLoadFunctor_[0-9]+]]()
// CHECK: _MoggInputLoadFusionTensor[{{.*}}]([[F0]],

// CHECK: struct [[FN0]](InputLoadFusion, TrivialRegisterPassable):
// No `var` field between the struct and its `__call__`: nothing is captured.
// CHECK-NOT: var
// CHECK: def __call__[
// CHECK: return rebind[SIMD[dtype, simd_width]](SIMD[DType.int32, simd_width](7))

mo.graph @concat_constant(%arg: !mo.tensor<[2], si32>) -> (!mo.tensor<[3], si32>) {
  %c = mo.constant <#M.dense_array<7> : tensor<1xsi32>> : !mo.tensor<[1], si32>
  %out = mo.concat[0](%c, %arg) : (
    !mo.tensor<[1], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[3], si32>
  mo.output %out : !mo.tensor<[3], si32>
}
