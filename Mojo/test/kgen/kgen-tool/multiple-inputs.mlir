// RUN: not kgen --emit=llvm %s %s 2>&1 | FileCheck %s -DFILE=%s

// CHECK: kgen: error: expected one input file, but got 2: [[FILE]], [[FILE]]

kgen.generator export @some_func(%arg0: f32) -> f32 {
  hlcf.return %arg0 : f32
}
