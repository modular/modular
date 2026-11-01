// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_lib default-device-label=gpu compiled-framework-label=mof" --allow-unregistered-dialect
// RUN: mt --execute %t.mlir --result-output-style=full --input-shapes=4 --model-inputs=ones | FileCheck %s

// End-to-end proof of the dense GPU trait-argument emit path (Slice 3 /
// Phase 4). `trait_negate` is written with the composable tensor-argument
// traits (`Some[Output & TileTensorable]`); for a GPU device the graph compiler
// emits the same inline `_MoggDenseTensorArg` but with `device_passable = True`,
// so the value crosses to the device and `_to_device_type` projects it to a
// device `TileTensor`. The kernel's GPU arm enqueues `_gpu_negate_kernel`. With
// an all-ones input the negate kernel must return all -1.

// The input/output must live on the GPU for a real device-launch kernel, so the
// graph transfers `%x` to the device and the result back (mirroring the working
// `ndbuffer_matmul` GPU e2e). Without the transfers the GC allocates the buffers
// on the host and the device kernel writes unreachable host memory (-> zeros).
// CHECK: 'trait_negate_graph' returned tensor<4xf32> [-1.000000e+00, -1.000000e+00, -1.000000e+00, -1.000000e+00]
mo.graph @trait_negate_graph(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
    %ch0 = mo.chain.create()
    %x_gpu, %ch1 = mo.transfer[%ch0] %x : !mo.tensor<[4], f32> to <"gpu", 0>
    %z_gpu = mo.custom [symbol = "trait_negate", device = #M.device_ref<"gpu", 0>] (%x_gpu) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    %z, %ch2 = mo.transfer[%ch1] %z_gpu : !mo.tensor<[4], f32, gpu:0> to <"cpu", 0>
    mo.output %z : !mo.tensor<[4], f32>
}
