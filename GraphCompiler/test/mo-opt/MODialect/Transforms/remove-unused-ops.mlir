// RUN: mo-opt %s --remove-unused-ops | FileCheck %s

mo.graph @unused_all_reduce<total_seq_len>(%arg0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, %arg1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, %arg2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, %arg3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, %arg4: !mo.buffer<[537919488], ui8, gpu:0>, %arg5: !mo.buffer<[537919488], ui8, gpu:1>, %arg6: !mo.buffer<[537919488], ui8, gpu:2>, %arg7: !mo.buffer<[537919488], ui8, gpu:3>, %arg8: !mo.chain, %arg9: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, %arg10: !mo.tensor<[16384], bf16, gpu:0>, %arg11: !mo.tensor<[], bf16>, %arg12: !mo.tensor<[], bf16>) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>) {
  %output:4, %outChain = mo.distributed.allreduce.sum(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)
  %0 = mo.add(%arg0, %arg0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
  mo.output %0 : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
}
// COM: Remove the unused allreduce.sum op
// CHECK-LABEL: mo.graph @unused_all_reduce
// CHECK-NEXT:  mo.add
// CHECK-NEXT:  mo.output


mo.graph @unused_all_reduce_inputs_outputs<total_seq_len>(%arg0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, %arg1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, %arg2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, %arg3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, %arg4: !mo.buffer<[537919488], ui8, gpu:0>, %arg5: !mo.buffer<[537919488], ui8, gpu:1>, %arg6: !mo.buffer<[537919488], ui8, gpu:2>, %arg7: !mo.buffer<[537919488], ui8, gpu:3>, %arg8: !mo.chain, %arg9: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, %arg10: !mo.tensor<[16384], bf16, gpu:0>, %arg11: !mo.tensor<[], bf16>, %arg12: !mo.tensor<[], bf16>) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>) {
  %0 = mo.mul(%arg0, %arg0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
  %output:4, %outChain = mo.distributed.allreduce.sum(%0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)
  %1 = mo.mul(%output#0, %output#0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
  %2 = mo.add(%arg0, %arg0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
  mo.output %2 : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
}
// COM: Remove the unused allreduce.sum op as well as the 2 mo.mul applied to its I/Os.
// CHECK-LABEL: mo.graph @unused_all_reduce_inputs_outputs
// CHECK-NEXT:  mo.add
// CHECK-NEXT:  mo.output

mo.graph @used_all_reduce_chain<total_seq_len>(%arg0: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, %arg1: !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, %arg2: !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, %arg3: !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, %arg4: !mo.buffer<[537919488], ui8, gpu:0>, %arg5: !mo.buffer<[537919488], ui8, gpu:1>, %arg6: !mo.buffer<[537919488], ui8, gpu:2>, %arg7: !mo.buffer<[537919488], ui8, gpu:3>, %arg8: !mo.chain, %arg9: !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, %arg10: !mo.tensor<[16384], bf16, gpu:0>, %arg11: !mo.tensor<[], bf16>, %arg12: !mo.tensor<[], bf16>) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain) {
  %output:4, %outChain = mo.distributed.allreduce.sum(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) : (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.buffer<[537919488], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:1>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:2>, !mo.tensor<[total_seq_len, 16384], bf16, gpu:3>, !mo.chain)
  %0 = mo.add(%arg0, %arg0) : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>
  mo.output %0, %outChain : !mo.tensor<[total_seq_len, 16384], bf16, gpu:0>, !mo.chain
}
// COM: All reduce can't be removed because the chain is used.
// CHECK-LABEL: mo.graph @used_all_reduce_chain
// CHECK-NEXT:  mo.distributed.allreduce.sum

mo.graph @unused_scatter(%arg0: !mo.tensor<[2, 3], f32, gpu:0>, %arg1: !mo.tensor<[2, 3], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %chain: !mo.chain, %arg5: !mo.tensor<[2, 3], f32, gpu:0>) -> (!mo.tensor<[2, 3], f32, gpu:0>) {
  %output:2, %outChain = mo.distributed.scatter(%arg0, %arg1, %sig0, %sig1, %chain) [root = 0] : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  %0 = mo.add(%arg5, %arg5) : !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %0 : !mo.tensor<[2, 3], f32, gpu:0>
}
// COM: Remove the unused scatter op
// CHECK-LABEL: mo.graph @unused_scatter
// CHECK-NEXT:  mo.add
// CHECK-NEXT:  mo.output

mo.graph @unused_print(%arg0: !mo.tensor<[4, 5], f32>) {
  %ch0 = mo.chain.create()
  mo.debug.tensor.print(%ch0, %arg0) [label = "label"] : !mo.tensor<[4, 5], f32>
  mo.output
}
// COM: TODO: Removing any mutable ops without uses isn't supported yet.
// CHECK-LABEL: mo.graph @unused_print
// CHECK:       mo.debug.tensor.print
