// RUN: mo-opt --gc-pipeline %s | FileCheck --check-prefix=CHECK-IR %s

// CHECK-IR-LABEL: mef.callable @graph$exec
// CHECK-IR: %{{.*}} = "mgp.generic.execute"{{.*}}mo.distributed.allreduce.sum{{.*}}
//
mo.graph @graph<total_seq_len>(%in0: !mo.tensor<[total_seq_len, 3], f32, gpu:0>, %in1: !mo.tensor<[total_seq_len, 3], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[total_seq_len, 3], f32, gpu:0>, !mo.tensor<[total_seq_len, 3], f32, gpu:1>) {
  %ch = mo.chain.create()
  %out:2, %outChain = mo.distributed.allreduce.sum(%in0, %in1, %sig0, %sig1, %ch) : (!mo.tensor<[total_seq_len, 3], f32, gpu:0>, !mo.tensor<[total_seq_len, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[total_seq_len, 3], f32, gpu:0>, !mo.tensor<[total_seq_len, 3], f32, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1 : !mo.tensor<[total_seq_len, 3], f32, gpu:0>, !mo.tensor<[total_seq_len, 3], f32, gpu:1>
}
