//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// Demonstrates device graph capture and replay using the MAX C API.
///
/// Device graph capture (e.g. CUDA graphs) records a model execution into a
/// replayable graph, eliminating per-launch overhead on subsequent runs. This
/// is useful for latency-sensitive inference where the same model is executed
/// repeatedly with the same input shapes.
///
/// Workflow:
///   1. Load a compiled model (MEF) and create input tensors on the GPU.
///   2. Capture: run the model once, recording the execution as a device graph.
///      This returns output tensors that will be updated in-place on each
///      replay.
///   3. Replay: re-execute the captured graph with near-zero launch overhead.
///      Read results from the same output tensors returned during capture.
///   4. Debug verify (optional): run eagerly and compare the kernel launch
///   trace
///      against the captured graph to check correctness.
///
//===----------------------------------------------------------------------===//

#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/device.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void printTensor(const float *data, size_t n) {
  printf("[");
  for (size_t i = 0; i < n; ++i) {
    printf("%.1f", data[i]);
    if (i < n - 1)
      printf(", ");
  }
  printf("]\n");
}

/// Helper: check status, print message, and jump to cleanup label on error.
#define CHECK(status, msg, label)                                              \
  do {                                                                         \
    if (M_isError(status)) {                                                   \
      printf("Error: %s: %s\n", msg, M_getError(status));                      \
      result = EXIT_FAILURE;                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0)

int main() {
  printf("=== MAX C API: Device Graph Capture & Replay ===\n\n");

  int result = EXIT_SUCCESS;
  M_Status *status = M_newStatus();
  M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();

  // --- Set up runtime with host + accelerator devices ---

  if (M_getAcceleratorCount() == 0) {
    printf("No accelerator available. This example requires a GPU.\n");
    result = EXIT_FAILURE;
    goto cleanup_config;
  }

  M_Device *host = M_newDevice(M_HOST, 0, status);
  CHECK(status, "creating host device", cleanup_config);
  M_runtimeConfigAddDevice(runtimeConfig, host);

  M_Device *gpu = M_newDevice(M_ACCELERATOR, 0, status);
  CHECK(status, "creating accelerator device", cleanup_host);
  M_runtimeConfigAddDevice(runtimeConfig, gpu);

  M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
  CHECK(status, "creating runtime context", cleanup_gpu);

  printf("Using device: %s\n\n", M_getDeviceLabel(gpu));

  // --- Load compiled model from MEF ---

  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, "graph.mef");

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  CHECK(status, "compiling model", cleanup_context);

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  CHECK(status, "initializing model", cleanup_compiled);

  // --- Create input tensors on the GPU ---
  //
  // Graph capture works with GPU-resident tensors directly. We borrow host
  // data into a tensor map, extract individual tensors, then copy them to
  // the accelerator.

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  int64_t shape[1] = {8};

  M_TensorSpec *hostSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_TensorSpec *hostSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", host);

  M_AsyncTensorMap *hostMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(hostMap, vector1, hostSpec1, status);
  CHECK(status, "borrowing vector1", cleanup_host_map);
  M_borrowTensorInto(hostMap, vector2, hostSpec2, status);
  CHECK(status, "borrowing vector2", cleanup_host_map);

  M_AsyncTensor *hostTensor1 = M_getTensorByNameFrom(hostMap, "input0", status);
  CHECK(status, "getting host tensor1", cleanup_host_map);
  M_AsyncTensor *hostTensor2 = M_getTensorByNameFrom(hostMap, "input1", status);
  CHECK(status, "getting host tensor2", cleanup_host_tensors);

  M_AsyncTensor *gpuInput1 = M_copyTensorToDevice(hostTensor1, gpu, status);
  CHECK(status, "copying tensor1 to GPU", cleanup_host_tensors);
  M_AsyncTensor *gpuInput2 = M_copyTensorToDevice(hostTensor2, gpu, status);
  CHECK(status, "copying tensor2 to GPU", cleanup_gpu_inputs);

  // --- Step 1: Capture ---
  //
  // M_captureModelSync runs the model once and records the execution as a
  // device graph. The graph_key (a uint64_t) identifies this captured graph
  // for later replay. The function returns output tensors that are updated
  // in-place on each subsequent replay.
  //
  // Note: Device graph capture requires CUDA or HIP. If the accelerator does
  // not support it (e.g. Apple Metal on macOS), capture will fail.

  printf("Step 1: Capturing model execution as device graph...\n");

  uint64_t graphKey = 1;
  M_AsyncTensor *inputs[2] = {gpuInput1, gpuInput2};
  size_t numOutputs = 0;

  M_AsyncTensor **capturedOutputs = M_captureModelSync(
      context, model, &graphKey, 1, inputs, 2, &numOutputs, status);
  if (M_isError(status)) {
    printf("Device graph capture is not supported on this accelerator.\n");
    printf("This feature requires a CUDA or HIP GPU.\n");
    goto cleanup_gpu_inputs;
  }

  printf("  Captured graph with key=%llu, %zu output(s)\n\n",
         (unsigned long long)graphKey, numOutputs);

  // Read capture-time results (copy to host for printing).
  M_AsyncTensor *outputOnHost =
      M_copyTensorToDevice(capturedOutputs[0], host, status);
  CHECK(status, "copying output to host", cleanup_captured);

  const float *data = (const float *)M_getTensorData(outputOnHost);
  printf("  Input 1: ");
  printTensor(vector1, 8);
  printf("  Input 2: ");
  printTensor(vector2, 8);
  printf("  Output:  ");
  printTensor(data, 8);
  M_freeTensor(outputOnHost);

  // --- Step 2: Replay ---
  //
  // M_replayModelSync re-executes the captured graph with near-zero launch
  // overhead. The same input tensors must be used (same buffer addresses).
  // Results appear in the output tensors returned by M_captureModelSync.

  printf("\nStep 2: Replaying captured graph...\n");

  M_replayModelSync(context, model, &graphKey, 1, inputs, 2, status);
  CHECK(status, "replaying device graph", cleanup_captured);

  outputOnHost = M_copyTensorToDevice(capturedOutputs[0], host, status);
  CHECK(status, "copying replay output to host", cleanup_captured);
  data = (const float *)M_getTensorData(outputOnHost);
  printf("  Replay output: ");
  printTensor(data, 8);
  M_freeTensor(outputOnHost);

  // --- Step 3: Debug verify replay (optional) ---
  //
  // M_debugVerifyReplayModelSync runs the model eagerly and compares the
  // kernel launch trace against the captured graph. Use this during
  // development to verify that captured graphs remain correct.

  printf("\nStep 3: Verifying captured graph matches eager execution...\n");

  M_debugVerifyReplayModelSync(context, model, &graphKey, 1, inputs, 2, status);
  CHECK(status, "debug verify replay", cleanup_captured);

  printf("  Verification passed!\n");

  printf("\nDevice graph capture and replay completed successfully.\n");

  // --- Cleanup ---

cleanup_captured:
  for (size_t i = 0; i < numOutputs; ++i)
    M_freeTensor(capturedOutputs[i]);
  free(capturedOutputs);
cleanup_gpu_inputs:
  M_freeTensor(gpuInput2);
  M_freeTensor(gpuInput1);
cleanup_host_tensors:
  M_freeTensor(hostTensor2);
  M_freeTensor(hostTensor1);
cleanup_host_map:
  M_freeAsyncTensorMap(hostMap);
  M_freeTensorSpec(hostSpec2);
  M_freeTensorSpec(hostSpec1);
  M_freeModel(model);
cleanup_compiled:
  M_freeCompiledModel(compiledModel);
cleanup_context:
  M_freeRuntimeContext(context);
cleanup_gpu:
  M_freeDevice(gpu);
cleanup_host:
  M_freeDevice(host);
cleanup_config:
  M_freeRuntimeConfig(runtimeConfig);
  M_freeStatus(status);

  return result;
}
