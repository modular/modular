//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "max/c/model.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "Support/AssertStream.h"
#include "Support/ErrorOr.h"
#include "max/max_license/API/c/CompiledModels.h"
#include "max/max_license/API/c/EngineContext.h"
#include "max/max_license/API/c/ModelExecution.h"
#include "max/max_license/API/c/Telemetry.h"
#include "max/max_license/API/c/detail/helpers.h"
#include "max/max_license/API/c/detail/types.h"

#include <cstdlib>
#include <optional>

using namespace M;

MODULAR_API_EXPORT M_CompileConfig *M_newCompileConfig() {
  auto *config = new M_CompileConfig();
  config->config = CompilationRequest();
  return config;
}

MODULAR_API_EXPORT void M_setModelPath(M_CompileConfig *compileConfig,
                                       const char *path) {
  compileConfig->config.setModelPath(std::filesystem::path(path));
}

MODULAR_API_EXPORT M_AsyncCompiledModel *
M_compileModel(const M_RuntimeContext *context, M_CompileConfig **compileConfig,
               M_Status *status) {
  if (!compileConfig || !*compileConfig)
    return nullptr;

  // Take ownership of config and reset the value of pointer.
  CompilationRequest config = std::move((*compileConfig)->config);
  delete *compileConfig;
  *compileConfig = nullptr;

  if (!context || !status)
    return nullptr;

  const Engine::Context &runtimeContext = context->context;

  AsyncRT::AnyAsyncValueRef compiledModel =
      runtimeContext.compileModel(std::move(config));
  if (compiledModel.isError()) {
    auto &telemetryCtx =
        *runtimeContext.maxContext->get<GML::MaxEngineTelemetryContext>();
    Error error = compiledModel.getDiagnostic().getMessage().copy();
    Telemetry::emitCompileInfo("compile.failure", telemetryCtx,
                               std::optional{error.copy()});
    M::Detail::setError(status, error.copy());
    return nullptr;
  }
  M::Detail::setOK(status);
  auto *compiledModelPtr = new M_AsyncCompiledModel();
  compiledModelPtr->valueOrError = compiledModel.copy();
  return compiledModelPtr;
}

MODULAR_API_EXPORT M_AsyncCompiledModel *
M_compileModelSync(const M_RuntimeContext *context,
                   M_CompileConfig **compileConfig, M_Status *status) {
  M_AsyncCompiledModel *compiledModel =
      M_compileModel(context, compileConfig, status);
  if (compiledModel)
    M_waitForCompilation(compiledModel, status);

  return compiledModel;
}

MODULAR_API_EXPORT M_AsyncModel *
M_initModel(const M_RuntimeContext *context,
            const M_AsyncCompiledModel *compiledModel,
            const M_WeightsRegistry *weightsRegistry, M_Status *status) {
  if (!context || !compiledModel || !compiledModel->valueOrError || !status)
    return nullptr;
  const Engine::Context &runtimeContext = context->context;
  auto *initializedModel = new M_AsyncModel();
  // The C API exposes only the first model from the compiled artifact.
  auto modelsAsync = runtimeContext.initModel(
      compiledModel->valueOrError.copy(),
      weightsRegistry ? weightsRegistry->weights : WeightsRegistryRef());
  // Await the async result.  The actual model setup runs on a cpuDevice worker
  // thread (dispatched inside initModel), so blocking here is safe.
  auto firstModel =
      AnyAsyncValueRef::allocate<std::unique_ptr<ExecutableModel>>(
          runtimeContext.getCPUDevice());
  if (auto err = Engine::Context::await(modelsAsync.copy())) {
    M::Detail::setError(status, err.takeError());
    std::move(firstModel).setToError(modelsAsync.takeDiagnostic());
  } else {
    M::Detail::setOK(status);
    firstModel.copy().emplace<std::unique_ptr<ExecutableModel>>(
        std::move(modelsAsync.get().front()));
  }
  initializedModel->modelOrError = AsyncExecutableModel{std::move(firstModel)};
  return initializedModel;
}

MODULAR_API_EXPORT void
M_waitForCompilation(M_AsyncCompiledModel *compiledModel, M_Status *status) {
  if (!compiledModel || !compiledModel->valueOrError || !status)
    return;

  ErrorOrSuccess modelOr =
      Engine::Context::await(compiledModel->valueOrError.copy());
  if (modelOr.isError())
    M::Detail::setError(status, modelOr.takeError());
  else
    M::Detail::setOK(status);
}

MODULAR_API_EXPORT void M_waitForModel(M_AsyncModel *model, M_Status *status) {
  if (!model || !model->modelOrError || !status)
    return;

  ErrorOrSuccess modelOr = Engine::Context::await(model->modelOrError.copy());
  if (modelOr.isError())
    M::Detail::setError(status, modelOr.takeError());
  else
    M::Detail::setOK(status);
}

MODULAR_API_EXPORT M_AsyncTensorMap *
M_executeModelSync(const M_RuntimeContext *context,
                   M_AsyncModel *initializedModel, M_AsyncTensorMap *inputs,
                   M_Status *status) {
  const Engine::Context &runtimeContext = context->context;

  auto &modelOrError = initializedModel->modelOrError;
  AsyncRT::await(modelOrError);
  if (modelOrError.isError()) {
    M::Detail::setError(status,
                        modelOrError.getDiagnostic().getMessage().copy());
    return nullptr;
  }
  std::unique_ptr<ExecutableModel> &executableModel = modelOrError.get();
  if (!executableModel) {
    M::Detail::setError(status,
                        Error::getStaticString("Executable model not ready"));
    return nullptr;
  }
  auto *model = executableModel->getModel();
  auto inputNames = model->getInputNames();

  // Prepare inputs in MEF order by looking them up by name
  llvm::SmallVector<AnyAsyncValueRef> inputValues;
  inputValues.reserve(inputNames.size());
  for (const auto &inputName : inputNames) {
    auto entry = inputs->indices.find(inputName);
    if (entry == inputs->indices.end()) {
      M::Detail::setError(status, Error("Required input named '" + inputName +
                                        "' not provided."));
      return nullptr;
    }
    auto index = entry->second;
    auto storage = inputs->tensors[index];
    if (!storage) {
      M::Detail::setError(status, Error("Input named '" + inputName +
                                        "' has no backing storage."));
      return nullptr;
    }
    auto inputSpec = model->getInputSpecByName(inputName);
    ASSERT_STREAM(!inputSpec.isError(), "no input spec for required input");

    // Validate input spec using shared validation logic
    ErrorOrSuccess validationResult =
        M::GraphCompiler::Driver::validateInputSpec(storage->spec, *inputSpec);
    if (validationResult.isError()) {
      M::Detail::setError(status, Error("input '" + inputName + "': " +
                                        validationResult.takeError().get()));
      return nullptr;
    }
    if (storage->device->ref != model->getInputDevices()[index]) {
      M::Detail::setError(
          status, Error("input '" + inputName + "' on incorrect device"));
      return nullptr;
    }

    inputValues.push_back(M::GraphCompiler::Driver::prepareInput(
        runtimeContext.getCPUDevice(), storage->buffer, storage->spec));
  }

  // Execute using the driver path
  auto resultsOr = M::GraphCompiler::Driver::executeFromDriver(
      runtimeContext, modelOrError.copy(), inputValues);

  if (resultsOr.isError()) {
    M::Detail::setError(status, resultsOr.takeError());
    return nullptr;
  }

  M::Detail::setOK(status);

  // Convert results to M_AsyncTensorMap
  auto execResults = resultsOr.takeValue();

  llvm::SmallVector<std::shared_ptr<M_TensorStorage>> outputTensors;
  outputTensors.reserve(execResults.size());
  llvm::SmallDenseMap<M::Driver::DeviceBuffer *,
                      std::shared_ptr<M_TensorStorage>, 8>
      storageCache;
  for (const auto &result : execResults) {
    // Aliasing outputs share one buffer and must share one storage, so the
    // C API hands out the same pointer for both.
    auto [it, inserted] =
        storageCache.try_emplace(result.buffer.getPointer(), nullptr);
    if (inserted)
      it->getSecond() = std::make_shared<M_TensorStorage>(
          M_TensorStorage{result.buffer, result.spec, result.device});
    outputTensors.push_back(it->getSecond());
  }

  // Get output names from the model (already awaited above)
  auto outputNames = model->getOutputNames();
  llvm::StringMap<size_t> indices;
  for (const auto &entry : llvm::enumerate(outputNames))
    indices[entry.value()] = entry.index();

  return new M_AsyncTensorMap{std::move(outputTensors), std::move(indices),
                              &runtimeContext};
}

/// Collect the device buffers backing an array of M_AsyncTensor pointers.
static ErrorOr<llvm::SmallVector<ModelInputBuffer>>
collectInputBuffers(M_AsyncTensor *const *inputs, size_t numInputs) {
  llvm::SmallVector<ModelInputBuffer> buffers;
  buffers.reserve(numInputs);
  for (size_t i = 0; i < numInputs; ++i) {
    if (!inputs[i] || !inputs[i]->storage)
      return Error("input tensor " + std::to_string(i) + " is invalid");
    buffers.push_back({inputs[i]->storage->buffer, inputs[i]->storage->spec});
  }
  return buffers;
}

MODULAR_API_EXPORT M_AsyncTensor **
M_captureModelSync(const M_RuntimeContext *context,
                   M_AsyncModel *initializedModel, const uint64_t *graphKeys,
                   size_t numGraphKeys, M_AsyncTensor *const *inputs,
                   size_t numInputs, size_t *numOutputs, M_Status *status) {
  if (!context || !initializedModel || !graphKeys || !inputs || !numOutputs) {
    if (status)
      M::Detail::setError(
          status, Error::getStaticString("NULL argument passed to capture"));
    return nullptr;
  }
  if (!status)
    return nullptr;

  auto &modelOrError = initializedModel->modelOrError;
  AsyncRT::await(modelOrError);
  if (modelOrError.isError()) {
    M::Detail::setError(status,
                        modelOrError.getDiagnostic().getMessage().copy());
    return nullptr;
  }
  std::unique_ptr<ExecutableModel> &executableModel = modelOrError.get();
  if (!executableModel) {
    M::Detail::setError(status,
                        Error::getStaticString("Executable model not ready"));
    return nullptr;
  }

  auto buffersOr = collectInputBuffers(inputs, numInputs);
  if (buffersOr.isError()) {
    M::Detail::setError(status, buffersOr.takeError());
    return nullptr;
  }
  auto buffers = buffersOr.takeValue();
  ArrayRef<uint64_t> keys(graphKeys, numGraphKeys);

  auto resultOr = executableModel->capture(keys, buffers);
  if (resultOr.isError()) {
    M::Detail::setError(status, resultOr.takeError());
    return nullptr;
  }

  const Engine::Context &runtimeContext = context->context;
  SmallVector<AnyAsyncValueRef> results = resultOr.takeValue();
  auto execResultsOr = M::GraphCompiler::Driver::extractDriverResults(
      runtimeContext, initializedModel->modelOrError.copy(), results);
  if (execResultsOr.isError()) {
    M::Detail::setError(status, execResultsOr.takeError());
    return nullptr;
  }

  auto execResults = execResultsOr.takeValue();
  size_t count = execResults.size();
  *numOutputs = count;

  auto **outputArray =
      static_cast<M_AsyncTensor **>(malloc(count * sizeof(M_AsyncTensor *)));
  if (!outputArray) {
    M::Detail::setError(status,
                        Error::getStaticString("Failed to allocate outputs"));
    return nullptr;
  }

  // Handle aliased outputs (same pattern as M_executeModelSync).
  llvm::SmallDenseMap<M::Driver::DeviceBuffer *,
                      std::shared_ptr<M_TensorStorage>, 8>
      storageCache;
  for (size_t i = 0; i < count; ++i) {
    const auto &result = execResults[i];
    auto [it, inserted] =
        storageCache.try_emplace(result.buffer.getPointer(), nullptr);
    if (inserted)
      it->getSecond() = std::make_shared<M_TensorStorage>(
          M_TensorStorage{result.buffer, result.spec, result.device});
    outputArray[i] = new M_AsyncTensor{it->getSecond()};
  }

  M::Detail::setOK(status);
  return outputArray;
}

MODULAR_API_EXPORT void M_replayModelSync(const M_RuntimeContext *context,
                                          M_AsyncModel *initializedModel,
                                          const uint64_t *graphKeys,
                                          size_t numGraphKeys,
                                          M_AsyncTensor *const *inputs,
                                          size_t numInputs, M_Status *status) {
  if (!context || !initializedModel || !graphKeys || !inputs) {
    if (status)
      M::Detail::setError(
          status, Error::getStaticString("NULL argument passed to replay"));
    return;
  }
  if (!status)
    return;

  auto &modelOrError = initializedModel->modelOrError;
  AsyncRT::await(modelOrError);
  if (modelOrError.isError()) {
    M::Detail::setError(status,
                        modelOrError.getDiagnostic().getMessage().copy());
    return;
  }
  std::unique_ptr<ExecutableModel> &executableModel = modelOrError.get();
  if (!executableModel) {
    M::Detail::setError(status,
                        Error::getStaticString("Executable model not ready"));
    return;
  }

  auto buffersOr = collectInputBuffers(inputs, numInputs);
  if (buffersOr.isError()) {
    M::Detail::setError(status, buffersOr.takeError());
    return;
  }
  auto buffers = buffersOr.takeValue();
  ArrayRef<uint64_t> keys(graphKeys, numGraphKeys);

  auto result = executableModel->replay(keys, buffers);
  if (result.isError())
    M::Detail::setError(status, result.takeError());
  else
    M::Detail::setOK(status);
}

MODULAR_API_EXPORT void M_debugVerifyReplayModelSync(
    const M_RuntimeContext *context, M_AsyncModel *initializedModel,
    const uint64_t *graphKeys, size_t numGraphKeys,
    M_AsyncTensor *const *inputs, size_t numInputs, M_Status *status) {
  if (!context || !initializedModel || !graphKeys || !inputs) {
    if (status)
      M::Detail::setError(status, Error::getStaticString(
                                      "NULL argument passed to debug verify"));
    return;
  }
  if (!status)
    return;

  auto &modelOrError = initializedModel->modelOrError;
  AsyncRT::await(modelOrError);
  if (modelOrError.isError()) {
    M::Detail::setError(status,
                        modelOrError.getDiagnostic().getMessage().copy());
    return;
  }
  std::unique_ptr<ExecutableModel> &executableModel = modelOrError.get();
  if (!executableModel) {
    M::Detail::setError(status,
                        Error::getStaticString("Executable model not ready"));
    return;
  }

  auto buffersOr = collectInputBuffers(inputs, numInputs);
  if (buffersOr.isError()) {
    M::Detail::setError(status, buffersOr.takeError());
    return;
  }
  auto buffers = buffersOr.takeValue();
  ArrayRef<uint64_t> keys(graphKeys, numGraphKeys);

  auto result = executableModel->debugVerifyReplay(keys, buffers);
  if (result.isError())
    M::Detail::setError(status, result.takeError());
  else
    M::Detail::setOK(status);
}

MODULAR_API_EXPORT void M_releaseCapturedGraphs(const M_RuntimeContext *context,
                                                M_AsyncModel *initializedModel,
                                                const uint64_t *graphKeys,
                                                size_t numGraphKeys,
                                                M_Status *status) {
  if (!context || !initializedModel || !graphKeys) {
    if (status)
      M::Detail::setError(status,
                          Error::getStaticString(
                              "NULL argument passed to releaseCapturedGraphs"));
    return;
  }
  if (!status)
    return;

  auto &modelOrError = initializedModel->modelOrError;
  AsyncRT::await(modelOrError);
  if (modelOrError.isError()) {
    M::Detail::setError(status,
                        modelOrError.getDiagnostic().getMessage().copy());
    return;
  }
  std::unique_ptr<ExecutableModel> &executableModel = modelOrError.get();
  if (!executableModel) {
    M::Detail::setError(status,
                        Error::getStaticString("Executable model not ready"));
    return;
  }

  ArrayRef<uint64_t> keys(graphKeys, numGraphKeys);
  auto result = executableModel->releaseCapturedGraphs(keys);
  if (result.isError())
    M::Detail::setError(status, result.takeError());
  else
    M::Detail::setOK(status);
}

MODULAR_API_EXPORT void M_freeModel(M_AsyncModel *model) { delete model; }

MODULAR_API_EXPORT void M_freeCompiledModel(M_AsyncCompiledModel *model) {
  delete model;
}

MODULAR_API_EXPORT void M_freeCompileConfig(M_CompileConfig *config) {
  if (config) {
    delete config;
  }
}
