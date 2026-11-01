// FIXME: KERN-1437
// RUN: mo-opt --gc-pipeline="compiled-framework-label=mof constant-fold=false default-device-label=gpu" %s -o %t
// RUN: mt --execute --default-device-label=gpu --soft-fusion-non-host-kernel-limit=3 --model-inputs=ones --result-output-style=compact %t | FileCheck %s

// Simply check it runs e2e without issues
// CHECK-LABEL: --- Running 'multihead_flash_attention':
// CHECK-NEXT: returned tensor

mo.graph @multihead_flash_attention(
    %q: !mo.tensor<[1, 1024, 32, 128], f32>,
    %k: !mo.tensor<[1, 1024, 32, 128], f32>,
    %v: !mo.tensor<[1, 1024, 32, 128], f32>,
    %mask: !mo.tensor<[1, 1024, 1024], f32>,
    %scalar: !mo.tensor<[], f32>
) -> !mo.tensor<[1, 1024, 32, 128], f32> {
  // Current kernel requires scalar normalizer to be on host
  %ch0 = mo.chain.create()
  %q_gpu, %ch1 = mo.transfer[%ch0] %q : !mo.tensor<[1, 1024, 32, 128], f32> to <"gpu", 0>
  %k_gpu, %ch2 = mo.transfer[%ch1] %k : !mo.tensor<[1, 1024, 32, 128], f32> to <"gpu", 0>
  %v_gpu, %ch3 = mo.transfer[%ch2] %v : !mo.tensor<[1, 1024, 32, 128], f32> to <"gpu", 0>
  %mask_gpu, %ch4 = mo.transfer[%ch3] %mask : !mo.tensor<[1, 1024, 1024], f32> to <"gpu", 0>
  %res_gpu = mo.composite.masked_flash_attention_gpu(%q_gpu, %k_gpu, %v_gpu, %mask_gpu, %scalar) : (
    !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>,
    !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>,
    !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>,
    !mo.tensor<[1, 1024, 1024], f32, gpu:0>,
    !mo.tensor<[], f32>
) -> !mo.tensor<[1, 1024, 32, 128], f32, gpu:0>
  %res, %ch5 = mo.transfer[%ch4] %res_gpu : !mo.tensor<[1, 1024, 32, 128], f32, gpu:0> to <"cpu", 0>
  mo.output %res : !mo.tensor<[1, 1024, 32, 128], f32>
}
