// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck --enable-var-scope %s

// CHECK-LABEL: mo.graph @buffer_type
// CHECK-SAME: {{.*}}: !mo.buffer<[10, 20], f32>,
// CHECK-SAME: {{.*}}: !mo.buffer<[?, ?], f32>,
// CHECK-SAME: {{.*}}: !mo.buffer<[D0, D1], f32>,
// CHECK-SAME: {{.*}}: !mo.buffer<[D0, ?, 10], f32>
mo.graph @buffer_type<D0, D1>(
  %b0: !mo.buffer<[10, 20], f32>,
  %b1: !mo.buffer<[?, ?], f32>,
  %b2: !mo.buffer<[D0, D1], f32>,
  %b3: !mo.buffer<[D0, ?, 10], f32>
) {
  mo.output
}

// CHECK-LABEL: mo.graph @mutable_load
// CHECK-SAME: ([[VARG0:[^ ]*]]: !mo.chain, [[VARG1:[^ ]*]]: !mo.buffer<[10, 20], f32>)
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG1]], [[VARG0]]) : (!mo.buffer<[10, 20], f32>, !mo.chain) -> (!mo.tensor<[10, 20], f32>, !mo.chain)
// CHECK-NEXT: mo.guard
// CHECK-NEXT: mo.output
mo.graph @mutable_load(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>) -> !mo.chain {
  %tensor, %outChain = mo.mutable.load(%buf, %ch) :
    (!mo.buffer<[10, 20], f32>, !mo.chain) ->
    (!mo.tensor<[10, 20], f32>, !mo.chain)

  // want to make sure the load doesn't get DCE'd
  %ch1 = mo.guard[%ch](%outChain) : !mo.chain

  mo.output %ch1 : !mo.chain
}


// CHECK-LABEL: mo.graph @mutable_store
// CHECK-SAME: ([[VARG0:[^ ]*]]: !mo.chain, [[VARG1:[^ ]*]]: !mo.buffer<[10, 20], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[10, 20], f32>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.mutable.store([[VARG1]], [[VARG2]], [[VARG0]]) : (!mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output
mo.graph @mutable_store(%ch: !mo.chain, %buf: !mo.buffer<[10, 20], f32>, %tensor: !mo.tensor<[10, 20], f32>) {
  %outChain = mo.mutable.store(%buf, %tensor, %ch) :
    (!mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>, !mo.chain) -> (!mo.chain)
  mo.output
}

// equivalent to torch.add(blhs, rhs, out=blhs)
// CHECK-LABEL: mo.graph @inplace_add
// CHECK-SAME: ([[VARG0:[^ ]*]]: !mo.chain, [[VARG1:[^ ]*]]: !mo.buffer<[10, 20], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[10, 20], f32>) {
// CHECK-NEXT: [[VOUTTENSOR:[^ ]*]], [[VOUTCHAIN:[^ ]*]] = mo.mutable.load([[VARG1]], [[VARG0]]) : (!mo.buffer<[10, 20], f32>, !mo.chain) -> (!mo.tensor<[10, 20], f32>, !mo.chain)
// CHECK-NEXT: [[V0:[^ ]*]] = mo.add([[VOUTTENSOR]], [[VARG2]]) : !mo.tensor<[10, 20], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.mutable.store([[VARG1]], [[V0]], [[VOUTCHAIN]]) : (!mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>, !mo.chain) -> !mo.chain
// CHECK-NEXT: mo.output
mo.graph @inplace_add(%ch0: !mo.chain, %blhs: !mo.buffer<[10, 20], f32>, %rhs: !mo.tensor<[10, 20], f32>) {

  %vblhs, %ch1 = mo.mutable.load(%blhs, %ch0) :
    (!mo.buffer<[10, 20], f32>, !mo.chain) ->
    (!mo.tensor<[10, 20], f32>, !mo.chain)

  %res = mo.add(%vblhs, %rhs) : !mo.tensor<[10, 20], f32>

  %ch2 = mo.mutable.store(%blhs, %res, %ch1) :
    (!mo.buffer<[10, 20], f32>, !mo.tensor<[10, 20], f32>, !mo.chain) -> (!mo.chain)

  mo.output
}
