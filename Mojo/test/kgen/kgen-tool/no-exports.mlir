// RUN: not kgen %s --emit=object 2>&1 | FileCheck %s
// CHECK: module does not `@export` any symbols or define a `main` function; nothing to codegen

kgen.generator @f(%arg: f32) -> f32 {
  hlcf.return %arg : f32
}
