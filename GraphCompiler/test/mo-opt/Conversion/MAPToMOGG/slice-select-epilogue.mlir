// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s

// A matmul read by two adds, one of them returned whole and as a strided
// window. The adds pair as siblings and the relu of the window joins them in a
// select, then output fusion moves the kernel into the matmul's epilogue. The
// select lowers in place to a `mogg.index.slice.select` that holds the
// window's store at the select's index. The matmul calls its epilogue only
// when told it does work, so the call must report epilogue fusion.

// CHECK-LABEL: mo.graph @matmul_window
// CHECK:       %[[WINDOW:[0-9]+]] = mogg.tensor.create(%{{.*}}) {{.*}}io = Output{{.*}}shapeAttr = #mosh<ape[4, 16]>
// CHECK:       %[[EPI:[0-9]+]] = mogg.tensor.create.epilogue %{{.*}} (%{{[a-z0-9]+}} : !mogg.simd<f32>, %[[IDX:[a-z0-9]+]] : !map.index<2>) capturing {{.*}}%[[CAP:[a-z0-9]+]] = %[[WINDOW]] : !mogg.tensor} ptr
// CHECK:       mogg.index.slice.select(%[[IDX]]) <inShape = #mosh<ape[8, 16]> : !mosh.ape, outShape = #mosh<ape[4, 16]> : !mosh.ape, staticStart = array<i64: 1, 0>, staticStep = array<i64: 2, 1>> : !map.index<2> {
// CHECK-NEXT:  ^bb0(%[[IOUT:[a-z0-9]+]]: !map.index<2>):
// CHECK:         mogg.tensor.store %[[CAP]][%[[IOUT]]]
// CHECK-NEXT:  }
// CHECK:       mogg.call.opaque @{{.*}}.MatmulCPU{{.*}} outputs(%[[EPI]] : !mogg.tensor) {{.*}}hasEpilogueFusion = true
mo.graph @matmul_window(%p: !mo.tensor<[8, 8], f32>, %q: !mo.tensor<[8, 16], f32>, %b: !mo.tensor<[8, 16], f32>, %c: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<8, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%p, %q) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 16], f32>) -> !mo.tensor<[8, 16], f32>
  %x = mo.add(%mm, %b) : !mo.tensor<[8, 16], f32>
  %z = mo.add(%mm, %c) : !mo.tensor<[8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[8, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 16], f32>
  mo.output %x, %z, %y : !mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// The same window over a dynamic dim, with a negative start. The select reads
// its in- and out-shapes at runtime, threaded into the epilogue as shape
// captures, and keeps the start as written.

// CHECK-LABEL: mo.graph @matmul_window_dynamic
// CHECK:       mogg.tensor.create.epilogue %{{.*}} (%{{[a-z0-9]+}} : !mogg.simd<f32>, %[[IDX:[a-z0-9]+]] : !map.index<2>) capturing {{.*}} shapes {%[[IN:[a-z0-9]+]] = %{{[0-9]+}}, %[[OUT:[a-z0-9]+]] = %{{[0-9]+}}}
// CHECK:       mogg.index.slice.select(%[[IDX]], %[[IN]], %[[OUT]]) <inShape = #mosh<ape[M, 16]> : !mosh.ape, outShape = #mosh<ape[H, 16]> : !mosh.ape, staticStart = array<i64: -8, 0>, staticStep = array<i64: 2, 1>> : !map.index<2> {
// CHECK:       mogg.call.opaque @{{.*}}.MatmulCPU{{.*}} hasEpilogueFusion = true
mo.graph @matmul_window_dynamic<M, H>(%p: !mo.tensor<[M, 8], f32>, %q: !mo.tensor<[8, 16], f32>, %b: !mo.tensor<[M, 16], f32>, %c: !mo.tensor<[M, 16], f32>, %stops: !mo.tensor<[2], si64>) -> (!mo.tensor<[M, 16], f32>, !mo.tensor<[M, 16], f32>, !mo.tensor<[H, 16], f32>) {
  %starts = mo.constant <#M.dense_array<-8, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%p, %q) : (!mo.tensor<[M, 8], f32>, !mo.tensor<[8, 16], f32>) -> !mo.tensor<[M, 16], f32>
  %x = mo.add(%mm, %b) : !mo.tensor<[M, 16], f32>
  %z = mo.add(%mm, %c) : !mo.tensor<[M, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[M, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[H, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[H, 16], f32>
  mo.output %x, %z, %y : !mo.tensor<[M, 16], f32>, !mo.tensor<[M, 16], f32>, !mo.tensor<[H, 16], f32>
}
