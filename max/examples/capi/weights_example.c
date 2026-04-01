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

/// Demonstrates loading a model with external weights via the C API.
///
/// This example loads a precompiled MEF file that expects a weight named
/// "weight" to be provided at runtime via the weights registry. The graph
/// computes: output = input * weight.

#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"
#include "max/c/weights.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  printf("Loading MEF with external weights...\n");

  int result = EXIT_SUCCESS;

  // Initialize MAX runtime
  M_Status *status = M_newStatus();
  M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();
  M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
  if (M_isError(status)) {
    printf("Error creating runtime context: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupRuntimeConfig;
  }

  M_Device *host = M_newDevice(M_HOST, 0, status);
  if (M_isError(status)) {
    printf("Error creating host device: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupContext;
  }

  // Load the compiled model from a MEF file
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, "weights_graph.mef");

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  if (M_isError(status)) {
    printf("Error compiling model: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupHost;
  }

  // Create the weights registry with model weights.
  // The graph expects a weight named "weight" with shape (4,) float32.
  // These values would typically be loaded from a file (e.g. safetensors).
  float weightData[4] = {2.0f, 3.0f, 4.0f, 5.0f};
  const char *weightNames[1] = {"weight"};
  const void *weightPtrs[1] = {weightData};

  M_WeightsRegistry *weights =
      M_newWeightsRegistry(weightNames, weightPtrs, 1, status);
  if (M_isError(status)) {
    printf("Error creating weights registry: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupCompiledModel;
  }

  // Initialize the model with the weights registry
  M_AsyncModel *model = M_initModel(context, compiledModel, weights, status);
  if (M_isError(status)) {
    printf("Error initializing model: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupWeights;
  }

  // Create input: [1, 2, 3, 4]
  float inputData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[1] = {4};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, inputData, inputSpec, status);
  if (M_isError(status)) {
    printf("Error creating input tensor: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupInputs;
  }

  // Execute: output = input * weight = [2, 6, 12, 20]
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  if (M_isError(status)) {
    printf("Error executing model: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupInputs;
  }

  // Read the output
  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  if (M_isError(status)) {
    printf("Error getting output: %s\n", M_getError(status));
    result = EXIT_FAILURE;
    goto cleanupOutputs;
  }

  const float *outputData = (const float *)M_getTensorData(outputTensor);
  size_t numElements = M_getTensorNumElements(outputTensor);

  printf("\nInput:   [");
  for (size_t i = 0; i < numElements; ++i)
    printf("%.1f%s", inputData[i], i < numElements - 1 ? ", " : "");
  printf("]\n");

  printf("Weight:  [");
  for (size_t i = 0; i < numElements; ++i)
    printf("%.1f%s", weightData[i], i < numElements - 1 ? ", " : "");
  printf("]\n");

  printf("Output:  [");
  for (size_t i = 0; i < numElements; ++i)
    printf("%.1f%s", outputData[i], i < numElements - 1 ? ", " : "");
  printf("]\n");

  float expected[4] = {2.0f, 6.0f, 12.0f, 20.0f};
  printf("Expected:[");
  for (size_t i = 0; i < 4; ++i)
    printf("%.1f%s", expected[i], i < 3 ? ", " : "");
  printf("]\n");

  int correct = 1;
  for (size_t i = 0; i < numElements; ++i) {
    if (outputData[i] != expected[i]) {
      correct = 0;
      break;
    }
  }

  if (correct)
    printf("\nSuccess! input * weight computed correctly.\n");
  else {
    printf("\nFailed - results don't match expected values.\n");
    result = EXIT_FAILURE;
  }

  // Clean up
  M_freeTensor(outputTensor);
cleanupOutputs:
  M_freeAsyncTensorMap(outputs);
cleanupInputs:
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec);
  M_freeModel(model);
cleanupWeights:
  M_freeWeightsRegistry(weights);
cleanupCompiledModel:
  M_freeCompiledModel(compiledModel);
cleanupHost:
  M_freeDevice(host);
cleanupContext:
  M_freeRuntimeContext(context);
cleanupRuntimeConfig:
  M_freeRuntimeConfig(runtimeConfig);
  M_freeStatus(status);

  return result;
}
