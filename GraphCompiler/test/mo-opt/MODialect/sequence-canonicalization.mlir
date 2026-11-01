// RUN: mo-opt --canonicalize %s | FileCheck %s
// RUN: mo-opt --canonicalize --mo-symbolic-optimize %s | FileCheck %s --check-prefix=SYMOPT

// Weight prep inside a side-stream body moves in front of the sequence, so it
// runs on the main chain and can be lifted to init. The matmul on the block
// argument stays on the side stream.

// CHECK-LABEL: mo.graph @hoist_weight_prep
// CHECK: %[[W:.*]] = mo.constant.external
// CHECK: %[[T:.*]], %{{.*}} = mo.transfer
// CHECK-SAME: %[[W]]
// CHECK: %[[C:.*]] = mo.concat[0] (%[[T]], %[[T]])
// CHECK: mo.sequence[1]
// CHECK-NEXT: mo.matmul(%{{.*}}, %[[C]])
// CHECK-NEXT: mo.yield
mo.graph @hoist_weight_prep(%x: !mo.tensor<[4, 8], f32, gpu:0>) -> !mo.tensor<[4, 8], f32, gpu:0> {
  %w = mo.constant.external [name = "w", align = 4, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[4, 8], f32>
  %root = mo.chain.create()
  %s = mo.sequence[1] (%x : !mo.tensor<[4, 8], f32, gpu:0>) (%arg) -> (!mo.tensor<[4, 8], f32, gpu:0>) {
    %wd, %ch = mo.transfer[%root] %w : !mo.tensor<[4, 8], f32> to <"gpu", 0>
    %cat = mo.concat[0](%wd, %wd) : (!mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:0>) -> !mo.tensor<[8, 8], f32, gpu:0>
    %m = mo.matmul(%arg, %cat) : (!mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[8, 8], f32, gpu:0>) -> !mo.tensor<[4, 8], f32, gpu:0>
    mo.yield %m : !mo.tensor<[4, 8], f32, gpu:0>
  }
  mo.output %s : !mo.tensor<[4, 8], f32, gpu:0>
}

// Work on a captured activation is not weight prep and stays in the body.

// CHECK-LABEL: mo.graph @keep_captured_activation
// CHECK: mo.sequence[1]
// CHECK-NEXT: mo.exp
// CHECK: mo.yield
mo.graph @keep_captured_activation(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %prod = mo.relu(%a) : !mo.tensor<[2, 2], f32, gpu:0>
  %s = mo.sequence[1] (%b : !mo.tensor<[2, 2], f32, gpu:0>) (%arg) -> (!mo.tensor<[2, 2], f32, gpu:0>) {
    %r = mo.exp(%prod) : !mo.tensor<[2, 2], f32, gpu:0>
    %o = mo.add(%r, %arg) : !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %o : !mo.tensor<[2, 2], f32, gpu:0>
  }
  mo.output %s : !mo.tensor<[2, 2], f32, gpu:0>
}

// Two tensor-parallel shards slice one weight in two side-stream bodies. The
// slices are hoisted, so merging them into a split cannot put the split in
// one body and its use in the other.

// SYMOPT-LABEL: mo.graph @sharded_weight
// SYMOPT-COUNT-2: mo.slice
// SYMOPT: mo.sequence[1]
// SYMOPT-NOT: mo.slice
// SYMOPT: mo.sequence[1]
// SYMOPT-NOT: mo.slice
// SYMOPT: mo.output
mo.graph @sharded_weight(%x0: !mo.tensor<[4, 8], f32, gpu:0>, %x1: !mo.tensor<[4, 8], f32, gpu:1>) -> (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[4, 4], f32, gpu:1>) {
  %w = mo.constant.external [name = "w", align = 4, device = #M.device_ref<"cpu", 0>] : !mo.tensor<[8, 8], f32>
  %root = mo.chain.create()
  %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop0 = mo.constant <#M.dense_array<8, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %start1 = mo.constant <#M.dense_array<0, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop1 = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %s0 = mo.sequence[1] (%x0 : !mo.tensor<[4, 8], f32, gpu:0>) (%arg) -> (!mo.tensor<[4, 4], f32, gpu:0>) {
    %w0 = mo.slice(%w, %start0, %stop0, %step) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 4], f32>
    %wd, %ch = mo.transfer[%root] %w0 : !mo.tensor<[8, 4], f32> to <"gpu", 0>
    %m = mo.matmul(%arg, %wd) : (!mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[8, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:0>
    mo.yield %m : !mo.tensor<[4, 4], f32, gpu:0>
  }
  %s1 = mo.sequence[1] (%x1 : !mo.tensor<[4, 8], f32, gpu:1>) (%arg) -> (!mo.tensor<[4, 4], f32, gpu:1>) {
    %w1 = mo.slice(%w, %start1, %stop1, %step) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[8, 4], f32>
    %wd, %ch = mo.transfer[%root] %w1 : !mo.tensor<[8, 4], f32> to <"gpu", 1>
    %m = mo.matmul(%arg, %wd) : (!mo.tensor<[4, 8], f32, gpu:1>, !mo.tensor<[8, 4], f32, gpu:1>) -> !mo.tensor<[4, 4], f32, gpu:1>
    mo.yield %m : !mo.tensor<[4, 4], f32, gpu:1>
  }
  mo.output %s0, %s1 : !mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[4, 4], f32, gpu:1>
}
