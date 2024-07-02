// RUN: kgen %s -xray-instrument -emit-llvm | FileCheck %s

// CHECK: define dso_local float @exp_f32(float noundef %0) #[[FNATTRS:.*]]
// CHECK: attributes #[[FNATTRS:.*]] = {{.*}} "function-instrument"="xray-always"

kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
