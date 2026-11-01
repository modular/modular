// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --num-runs=1 --result-output-style=compact %s | FileCheck %s

// RUN: mo-opt %s -o %t.mlir --mo-legalize-rmo-operators --mo-infer-layouts --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --allow-unregistered-dialect
// RUN: cat %t.mlir | FileCheck --check-prefix=CHECK-IR %s

// CHECK-LABEL: --- Running 'add_dev0':
// CHECK: 'add_dev0' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4])
mo.graph @add_dev0(%a : !mo.tensor<[4], f32>, %b : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %da, %ch1 = rmo.mo.transfer[%ch0] %a: !mo.tensor<[4], f32> to <"gpu", 0>
  %db, %ch2 = rmo.mo.transfer[%ch1] %b: !mo.tensor<[4], f32> to <"gpu", 0>
  %dc = mo.add (%da, %db) : !mo.tensor<[4], f32, gpu:0>
  %c, %ch3 = rmo.mo.transfer[%ch2] %dc: !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
  mo.output %c : !mo.tensor<[4], f32>
}

// CHECK-LABEL: --- Running 'add_dev1':
// CHECK: 'add_dev1' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4])
mo.graph @add_dev1(%a : !mo.tensor<[4], f32>, %b : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %da, %ch1 = rmo.mo.transfer[%ch0] %a: !mo.tensor<[4], f32> to <"gpu", 1>
  %db, %ch2 = rmo.mo.transfer[%ch1] %b: !mo.tensor<[4], f32> to <"gpu", 1>
  %dc = mo.add (%da, %db) : !mo.tensor<[4], f32, gpu:1>
  %c, %ch3 = mo.transfer[%ch2] %dc: !mo.tensor<[4], f32, gpu:1> to <"cpu", 0>
  mo.output %c : !mo.tensor<[4], f32>
}

// CHECK-LABEL: --- Running 'transfer_dev01':
// CHECK: 'transfer_dev01' returned tensor({{\[}}[1.0000, 1.0000, 1.0000, 1.0000]], dtype=f32, shape=[4])
// CHECK-IR-LABEL: mgp.model @transfer_dev01
// CHECK-IR: [[H2D_CHAIN:%[0-9]+]] = mgp.buffer.host_to_device
// CHECK-IR: [[SYNC_CHAIN:%[0-9]+]] = mgp.sync({{.*}}[[H2D_CHAIN]]
// CHECK-IR: [[CH_CREATE:%[0-9]+]] = grt.chain.create {{.*}}[[H2D_CHAIN]]
// CHECK-IR: [[BEFORE_WAIT:%[0-9]+]] = mgp.device_wait({{.*}}[[CH_CREATE]]{{.*}}) device <"gpu", 1>
// CHECK-IR: [[D2D_CHAIN:%[0-9]+]] = mgp.buffer.device_to_device {{.*}}[[BEFORE_WAIT]]{{.*}}: !mgp.buffer<<"gpu", 0>> -> !mgp.buffer<<"gpu", 1>>
// CHECK-IR: mgp.device_wait({{.*}}[[D2D_CHAIN]]{{.*}}) device <"gpu", 0>
mo.graph @transfer_dev01(%a : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %da1, %ch1 = rmo.mo.transfer[%ch0] %a: !mo.tensor<[4], f32> to <"gpu", 0>
  %da0, %ch2 = rmo.mo.transfer[%ch1] %da1: !mo.tensor<[4], f32, gpu:0> to <"gpu", 1>
  %out, %ch3 = rmo.mo.transfer[%ch2] %da0: !mo.tensor<[4], f32, gpu:1> to <"cpu", 0>
  mo.output %out : !mo.tensor<[4], f32>
}

// CHECK-LABEL: --- Running 'add_dev01':
// CHECK: 'add_dev01' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4])
// CHECK-IR-LABEL: mgp.model @add_dev01
// CHECK-IR: [[CH_KERNEL:%[0-9]+]] = mogg.routine.async
// CHECK-IR: [[CH_CREATE:%[0-9]+]] = grt.chain.create {{.*}}[[CH_KERNEL]]
// CHECK-IR: [[BEFORE_WAIT:%[0-9]+]] = mgp.device_wait({{.*}}[[CH_CREATE]]{{.*}}) device <"gpu", 1>
// CHECK-IR: [[D2D_CHAIN:%[0-9]+]] = mgp.buffer.device_to_device {{.*}}[[BEFORE_WAIT]]{{.*}}: !mgp.buffer<<"gpu", 0>> -> !mgp.buffer<<"gpu", 1>>
// CHECK-IR: mgp.device_wait({{.*}}[[D2D_CHAIN]]{{.*}}) device <"gpu", 0>
mo.graph @add_dev01(%a : !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %da0, %ch1 = rmo.mo.transfer[%ch0] %a: !mo.tensor<[4], f32> to <"gpu", 0>
  %dx0 = mo.add(%da0, %da0) : !mo.tensor<[4], f32, gpu:0>
  %dx1, %ch2 = rmo.mo.transfer[%ch1] %dx0: !mo.tensor<[4], f32, gpu:0> to <"gpu", 1>
  %out, %ch3 = rmo.mo.transfer[%ch2] %dx1: !mo.tensor<[4], f32, gpu:1> to <"cpu", 0>
  mo.output %out : !mo.tensor<[4], f32>
}

// CHECK-LABEL: --- Running 'add_dev01_large':
// CHECK: 'add_dev01_large' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, {{.*}}, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[1024])
mo.graph @add_dev01_large(%a : !mo.tensor<[1024], f32>) -> !mo.tensor<[1024], f32> {
  %ch0 = mo.chain.create()
  %da1, %ch1 = rmo.mo.transfer[%ch0] %a: !mo.tensor<[1024], f32> to <"gpu", 0>
  %dax = mo.add(%da1, %da1) : !mo.tensor<[1024], f32, gpu:0>
  %da0, %ch2 = rmo.mo.transfer[%ch1] %dax: !mo.tensor<[1024], f32, gpu:0> to <"gpu", 1>
  %out, %ch3 = rmo.mo.transfer[%ch2] %da0: !mo.tensor<[1024], f32, gpu:1> to <"cpu", 0>
  mo.output %out : !mo.tensor<[1024], f32>
}
