// RUN: mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that elementwise mo ops lower to map.iter.elementwise (not map.iter.opaque)
// when the advanced fusion pipeline is enabled.

mo.graph @add(
  %x: !mo.tensor<[8, 16, 128], f32>,
  %y: !mo.tensor<[8, 16, 128], f32>
) -> !mo.tensor<[8, 16, 128], f32> {
  %output = mo.add(%x, %y) : !mo.tensor<[8, 16, 128], f32>
  mo.output %output : !mo.tensor<[8, 16, 128], f32>
}

// An elementwise op runs deferred, the default, so the foreach prints no
// `dispatch`.
// CHECK-LABEL:   mo.graph @add
// CHECK-NOT:       map.iter.opaque
// CHECK:           map.iter.elementwise<[8, 16, 128]>{{$}}
// CHECK:             [[XV:%.*]] = map.load %{{.*}}[[[IDX:%.*]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK:             [[YV:%.*]] = map.load %{{.*}}[[[IDX]]] : !mo.tensor<[8, 16, 128], f32>
// CHECK:             map.call.elementwise @{{.+}}.Add([[IDX]], [[XV]], [[YV]])
// CHECK:             map.store %{{.*}}[[[IDX]]] = %{{.*}} : !mo.tensor<[8, 16, 128], f32>
