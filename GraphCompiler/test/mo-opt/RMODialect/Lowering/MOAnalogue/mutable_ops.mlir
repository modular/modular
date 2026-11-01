// RUN: mo-opt --split-input-file --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @mutable_load_ok<d0, d1, d2, d3>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VOUTTENSOR]]) [message ={{.*}}] : (!mo.tensor<[d0, d1], f32>) -> !mo.tensor<[d2, d3], f32>
// CHECK-NEXT: mo.output [[VARG0]], [[V1]] : !mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>
mo.graph @mutable_load_ok<d0, d1, d2, d3>(%buf: !bt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)

  mo.output %buf, %vbuf : !bt, !tt
}

// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!c   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @mutable_store_ok<d0, d1, d2, d3>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VARG1]]) [message ={{.*}}] : (!mo.tensor<[d2, d3], f32>) -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.mutable.store([[VARG0]], [[V1]], [[V0]]) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output [[VARG0]], [[VARG1]] : !mo.buffer<[d0, d1], f32>, !mo.tensor<[d2, d3], f32>
mo.graph @mutable_store_ok<d0, d1, d2, d3>(%buf: !bt, %tensor: !tt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %ch1 = rmo.mo.mutable.store(%buf, %tensor, %ch0) : (!bt, !tt, !c) -> (!c)

  mo.output %buf, %tensor : !bt, !tt
}

// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!ct   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_all_unk<d0, d1, d2, d3 -> [[D0:[^ ]*]], [[D1:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VOUTTENSOR]]) [message ={{.*}}] : (!mo.tensor<[d0, d1], f32>) -> !mo.tensor<[d2, d3], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.add([[VARG1]], [[V1]]) : !mo.tensor<[d2, d3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[V2]]) [message ={{.*}}] : (!mo.tensor<[d2, d3], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) [message ={{.*}}] : (!mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.mutable.store([[VARG0]], [[V4]], [[VOUTCHAIN]]) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output<[[D2]], [[D3]]> [[VARG0]], [[V3]] : !mo.buffer<[d0, d1], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>
mo.graph @add_all_unk<d0, d1, d2, d3 -> D0, D1>(%buf: !bt, %lhs: !tt) -> (!bt, !mo.tensor<[D0, D1], f32>)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = rmo.mo.add<() -> D2, D3>(%lhs, %vbuf) : (!tt, !tt) -> !mo.tensor<[D2, D3], f32>
  %ch2 = rmo.mo.mutable.store(%buf, %res1, %ch1) : (!bt, !mo.tensor<[D2, D3], f32>, !ct) -> (!ct)

  mo.output<D2, D3> %buf, %res1 : !bt, !mo.tensor<[D2, D3], f32>
}

// -----

!tt  = !mo.tensor<[d0, d1], f32>
!bt  = !mo.buffer<[d2, d3], f32>
!ct   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_buf_unk<d0, d1, d2, d3>([[VARG0:[^ ]*]]: !mo.buffer<[d2, d3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[d0, d1], f32>) -> (!mo.buffer<[d2, d3], f32>, !mo.tensor<[d0, d1], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d2, d3], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[d2, d3], f32>, !mo.chain) -> (!mo.tensor<[d2, d3], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VOUTTENSOR]]) [message ={{.*}}] : (!mo.tensor<[d2, d3], f32>) -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.add([[VARG1]], [[V1]]) : !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape([[V2]]) [message ={{.*}}] : (!mo.tensor<[d0, d1], f32>) -> !mo.tensor<[d2, d3], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.mutable.store([[VARG0]], [[V3]], [[VOUTCHAIN]]) : (!mo.buffer<[d2, d3], f32>, !mo.tensor<[d2, d3], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output [[VARG0]], [[V2]] : !mo.buffer<[d2, d3], f32>, !mo.tensor<[d0, d1], f32>
mo.graph @add_buf_unk<d0, d1, d2, d3>(%buf: !bt, %lhs: !tt) -> (!bt, !tt)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = rmo.mo.add(%lhs, %vbuf) : (!tt, !tt) -> !tt
  %ch2 = rmo.mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf, %res1 : !bt, !tt
}

// -----

!tt  = !mo.tensor<[d2, d3], f32>
!bt  = !mo.buffer<[d0, d1], f32>
!ct   = !mo.chain

// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_tensor_unk<d0, d1, d2, d3 -> [[D0:[^ ]*]], [[D1:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.buffer<[d0, d1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[d2, d3], f32>) -> (!mo.buffer<[d0, d1], f32>, !mo.tensor<[[[D0]], [[D1]]], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.chain.create([[VARG0]]) : !mo.buffer<[d0, d1], f32>
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG0]], [[V0]]) : (!mo.buffer<[d0, d1], f32>, !mo.chain) -> (!mo.tensor<[d0, d1], f32>, !mo.chain)
// CHECK-NEXT: [[V1:[^ ]*]] = rmo.rebind_tensor_shape([[VOUTTENSOR]]) [message ={{.*}}] : (!mo.tensor<[d0, d1], f32>) -> !mo.tensor<[d2, d3], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.add([[VARG1]], [[V1]]) : !mo.tensor<[d2, d3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = rmo.rebind_tensor_shape<() -> [[D2:[^ ]*]], [[D3:[^ ]*]]>([[V2]]) [message ={{.*}}] : (!mo.tensor<[d2, d3], f32>) -> !mo.tensor<[[[D2]], [[D3]]], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = rmo.rebind_tensor_shape([[V3]]) [message ={{.*}}] : (!mo.tensor<[[[D2]], [[D3]]], f32>) -> !mo.tensor<[d0, d1], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.mutable.store([[VARG0]], [[V4]], [[VOUTCHAIN]]) : (!mo.buffer<[d0, d1], f32>, !mo.tensor<[d0, d1], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output<[[D2]], [[D3]]> [[VARG0]], [[V3]] : !mo.buffer<[d0, d1], f32>, !mo.tensor<[[[D2]], [[D3]]], f32>
mo.graph @add_tensor_unk<d0, d1, d2, d3 -> D0, D1>(%buf: !bt, %lhs: !tt) -> (!bt, !mo.tensor<[D0, D1], f32>)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = rmo.mo.mutable.load(%buf, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = rmo.mo.add<() -> D2, D3>(%lhs, %vbuf) : (!tt, !tt) -> !mo.tensor<[D2, D3], f32>
  %ch2 = rmo.mo.mutable.store(%buf, %res1, %ch1) : (!bt, !mo.tensor<[D2, D3], f32>, !ct) -> (!ct)

  mo.output<D2, D3> %buf, %res1 : !bt, !mo.tensor<[D2, D3], f32>
}
