// RUN: mo-opt --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp %s | FileCheck %s

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!ct = !mo.chain

// CHECK-LABEL: mgp.model @unfused_mutable_load_store
// CHECK: execute
// CHECK-SAME: !mgp.opaque{{.*}}, [[buf:%.*]]: !mgp.tensor<2x2xf32, mut>, [[rhs:%.*]]: !mgp.tensor<2x2xf32>)
// CHECK: grt.chain.create ([[buf]]
// CHECK: [[ebuf:%.*]] = mgp.tensor.extract.buffer{{.*}} [[buf]]
// The check below corresponds to the (lowered) mo.mutable.load
// CHECK: mgp.buffer.device_to_device{{.*}}[[ebuf]]
// CHECK: mogg.routine.async
// CHECK: mogg.routine.async
// CHECK: mogg.routine.async
mo.graph @unfused_mutable_load_store(%buf: !bt, %rhs: !tt) -> !bt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %res2 = mo.negative(%res1) : !tt
  %ch4 = mo.mutable.store(%buf, %res2, %ch1) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf : !bt
}
