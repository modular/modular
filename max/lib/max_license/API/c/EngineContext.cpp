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
//
// This file contains the core logic for the engine context, which is the entry
// point to a max engine session.
//
//===----------------------------------------------------------------------===//

#include "max/max_license/API/c/EngineContext.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "GraphCompiler/Driver/Device/DeviceMap.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "Init/Init.h"
#include "Support/Context.h"
#include "Support/ML/DebugPrint.h"
#include "Support/Telemetry/Telemetry.h"
#include "max/c/types.h"
#include "max/max_license/API/c/EngineConfig.h"
#include "max/max_license/API/c/SourceModel.h"
#include "max/max_license/API/c/Telemetry.h"
#include "max/max_license/API/c/TensorMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"

#include <mutex>
#include <string>
#include <string_view>

using namespace M;
using namespace AsyncRT;

//===----------------------------------------------------------------------===//
// RuntimeContext initialization
//===----------------------------------------------------------------------===//

static CPUDeviceOptions
getOrCreateCPUDeviceOptions(const Engine::Config &config,
                            StringRef profileFilename) {
  AsyncRT::CPUDeviceOptions cpuDeviceOptions;
  cpuDeviceOptions.numThreads = config.getNumThreads();
  cpuDeviceOptions.maxThreads = config.getMaxThreads();
  cpuDeviceOptions.tcmallocAllocator =
      config.getAllocatorType() == (int)kCaching;
  cpuDeviceOptions.threadBusyWaitTime = config.getThreadBusyWaitTime().count();
  cpuDeviceOptions.profileFilename = profileFilename;
  cpuDeviceOptions.poolName = config.getContextName();
  if (cpuDeviceOptions.numThreads == 1)
    cpuDeviceOptions.withSingleThreaded();
  // Previously mainWillDonate was required to be false in order to avoid the
  // Runtime created by the Engine from overwriting the Runtime created by the
  // Mojo stdlib during startup, this is no longer the case. Though there is
  // still a question of whether the Engine should be initialized with
  // mainWillDonate as true or false, setting to true currently causes an issue
  // with integration with the Torch inductor. See GEX-3387 for details.
  cpuDeviceOptions.mainWillDonate = false;

  return cpuDeviceOptions;
}

static void setUnknownLocError(AnyAsyncValueRef &&asyncVal, Error error) {
  std::move(asyncVal).setToError(
      {std::move(error), UnknownLocationDecoder::getEncodedLocation()});
}

static ErrorOr<std::unique_ptr<ExecutableModel>>
createExecutableModel(const M::Model *model, CPUDevice &cpuDevice,
                      const Engine::Config &config,
                      const WeightsRegistry *weightsRegistry) {
  M::GraphCompiler::Driver::DeviceMap deviceMap;
  for (const auto &device : config.getDevices())
    deviceMap.insert({device.spec->ref, device.context});
  auto executableModelOr = model->setup(cpuDevice, weightsRegistry, deviceMap);
  if (executableModelOr.isError())
    return executableModelOr.takeError().getMessage().copy();

  auto executableModel = executableModelOr.takeValue();
  auto setupErrorOrSuccess = executableModel->waitForModelSetup();
  if (setupErrorOrSuccess.isError())
    return Error(Twine("Model setup failed, not ready for execution.  "
                       "Modular internal detail: ") +
                 setupErrorOrSuccess.getError());
  return std::move(executableModel);
}

static ErrorOr<ExecutableModels>
createExecutableModels(const CompiledModels &compiledModels,
                       CPUDevice &cpuDevice, const Engine::Config &config,
                       const WeightsRegistry *weightsRegistry) {
  if (compiledModels.getNumModels() == 0)
    return Error::getStaticString(
        "Compiled model does not contain any valid models");

  ExecutableModels results;
  results.reserve(compiledModels.getNumModels());

  for (const std::unique_ptr<const Model> &model : compiledModels) {
    ErrorOr<std::unique_ptr<ExecutableModel>> executableModelOr =
        createExecutableModel(model.get(), cpuDevice, config, weightsRegistry);
    if (executableModelOr.isError())
      return executableModelOr.takeError();
    results.push_back(executableModelOr.takeValue());
  }
  return results;
}

static ErrorOr<TensorMap>
executeExecutableModel(const ExecutableModel &executableModel,
                       TensorMap &inputMap, CPUDevice &cpuDevice) {
  auto inputNames = executableModel.getModel()->getInputNames();

  SmallVector<AnyAsyncValueRef> inputValuesInMEFIndexOrder;
  inputValuesInMEFIndexOrder.reserve(inputMap.tensors.size());

  // Iterating on names in MEF order here.
  for (auto inputName : inputNames) {
    auto it = inputMap.tensors.find(inputName);
    if (it == inputMap.tensors.end()) {
      return Error("The model requires an input tensor named '" + inputName +
                   "' but it was not provided.");
    }

    inputValuesInMEFIndexOrder.emplace_back(it->second.copy());
  }

  auto results = executableModel.execute(inputValuesInMEFIndexOrder);

  if (results.empty())
    return Error("Failed to execute model");

  if (auto err = AsyncRT::awaitOrError(results))
    return err.takeError();

  MutableArrayRef<AnyAsyncValueRef> resultCopy(results.data(), results.size());
  resultCopy = resultCopy.drop_front();
  llvm::SmallVector<AnyAsyncValueRef> outputTensorRefs;
  outputTensorRefs.reserve(resultCopy.size());
  for (auto &result : resultCopy) {
    outputTensorRefs.push_back(std::move(result));
  }

  AsyncRT::await(outputTensorRefs);

  llvm::StringMap<AnyAsyncValueRef> resultMap;
  auto outputNames = executableModel.getModel()->getOutputNames();
  for (auto [outIdx, tensor] : llvm::enumerate(outputTensorRefs)) {
    if (tensor.isError())
      return tensor.getDiagnostic().getMessage().copy();

    resultMap.insert({outputNames[outIdx], std::move(tensor)});
  }

  return TensorMap(std::move(resultMap), cpuDevice);
}

ErrorOr<Engine::Context>
Engine::Context::create(const M::Engine::Config &config) {
  std::string profileFilename =
      llvm::sys::Process::GetEnv("MODULAR_PROFILE_FILENAME").value_or("");

  ContextRef maxContext;
  if (ContextRef ctx = config.getMaxContext()) {
    maxContext = std::move(ctx);
  } else {
    Init::Options options;
    CPUDeviceOptions rtOptions =
        getOrCreateCPUDeviceOptions(config, profileFilename);
    options.withCPUDeviceOptions(rtOptions);
    options.withForceDisableCrashReporting(!config.isCrashLogEnabled());

    auto contextOr = Init::getOrCreateContext("max-engine", options);
    if (contextOr.isError())
      return contextOr.takeError();
    maxContext = contextOr.takeValue();
  }
  auto runtimeContext = Engine::Context(config, maxContext.copy());
  auto &maxTelemetryContext =
      runtimeContext.maxContext
          ->emplaceIfMissing<GML::MaxEngineTelemetryContext>(maxContext.copy());
  maxTelemetryContext.setAPILanguage(config.getAPILanguage().str());

  // Source of truth for the closed set of valid device labels. Downstream
  // consumers (notably the C-API M_HOST vs M_ACCELERATOR predicate in
  // M_getDeviceType / M_getDeviceTypeFromSpec) treat anything not rejected
  // here as a valid device, so this guard must stay in sync with the
  // kCPULabel / kGPULabel constants in Support/DeviceSpecs.h.
  for (const auto &device : config.getDevices()) {
    const DeviceSpec &spec = *device.spec;
    if (spec.ref.label != kCPULabel && spec.ref.label != kGPULabel) {
      return Error(Twine("unrecognized device '") + spec.ref.label + "'");
    }
    // else: no special extra device handling
  }
  return runtimeContext;
}

Engine::Context::Context(Context &&other)
    : maxContext(std::move(other.maxContext)), config(std::move(other.config)),
      cpuDevice(*maxContext->get<CPUDevice>()) {}

Engine::Context::Context(const M::Engine::Config &runtimeConfig,
                         M::ContextRef context)
    : maxContext(std::move(context)), config(runtimeConfig),
      cpuDevice(*maxContext->get<CPUDevice>()) {
  crashRecoveryContext.DumpStackAndCleanupOnFailure = true;
}

//===----------------------------------------------------------------------===//
// Model compilation, loading, and initialization
//===----------------------------------------------------------------------===//

AsyncValueRef<CompiledModels>
Engine::Context::compileModel(CompilationRequest config) const {
  auto compiledModelAsync = AsyncValueRef<CompiledModels>::allocate(cpuDevice);

  addTask(cpuDevice, [config = std::move(config),
                      compiledModelAsync = compiledModelAsync.copy(),
                      this]() mutable {
    auto compiledModelOr = compileModelSync(std::move(config));

    auto &telemetryCtx = *maxContext->get<GML::MaxEngineTelemetryContext>();

    if (compiledModelOr.isError()) {
      Error compileError = compiledModelOr.takeError();
      // FIXME: This potentially can contain absolute file paths in user
      // machine. We might need to scrub that.
      Telemetry::emitCompileInfo("compile.failure", telemetryCtx,
                                 compileError.copy());
      return setUnknownLocError(std::move(compiledModelAsync),
                                std::move(compileError));
    }

    Telemetry::emitCompileInfo("compile.success", telemetryCtx, {});
    std::move(compiledModelAsync).emplace(compiledModelOr.takeValue());
  });

  return compiledModelAsync;
}

static void telemetryEmitModelFormat(FileFormat format, ContextRef context) {
  auto telemetryContext = context->get<Telemetry::TelemetryContext>();
  Telemetry::emitCompiledModelType(*telemetryContext, format);
}

static void telemetryEmitPipelineName(std::string_view pipelineName,
                                      ContextRef context) {
  auto telemetryContext = context->get<Telemetry::TelemetryContext>();
  Telemetry::emitPipelineName("max.pipeline.run", *telemetryContext,
                              pipelineName);
}

/// Compiles a model. Functionally identical to `M_compileModel`, but blocks
/// on execution. Unlike `M_compileModel`, the user does not need to call
/// `M_waitForCompilation` to await a compilation result. Also does not use CAS
/// for kernel caching.
ErrorOr<CompiledModels>
Engine::Context::compileModelSync(CompilationRequest config) const {
  config.setDevices(this->config.getDevices());
  auto modelOr =
      CompiledModels::create(maxContext.copy(), config, this->config);
  if (modelOr.isError())
    return modelOr.takeError();

  if (auto pipelineName = config.getPipelineName(); pipelineName)
    telemetryEmitPipelineName(*pipelineName, maxContext);
  telemetryEmitModelFormat(modelOr.get().getSourceModel().format(), maxContext);

  return modelOr.takeValue();
}

ErrorOrSuccess Engine::Context::await(AnyAsyncValueRef value) {
  return AsyncRT::awaitOrError(value);
}

M::AsyncExecutableModel
Engine::Context::initModel(const M::Model *model,
                           const WeightsRegistryRef &weightsRegistry) const {
  using ModelType = std::unique_ptr<M::ExecutableModel>;
  auto result = AsyncValueRef<ModelType>::allocate(cpuDevice);

  // Run on a cpuDevice worker thread to ensure AsyncRT cpuDevice is set in TLS.
  // (MGP_RT_BufferBorrowV2, called during Model::setup, requires a valid
  // current cpuDevice.)
  addTask(cpuDevice, [model, weights = weightsRegistry.copy(),
                      result = result.copy(), this]() mutable {
    auto executableModelOr = createExecutableModel(
        model, cpuDevice, getConfig(), weights.getPointer());
    if (executableModelOr.isError())
      return setUnknownLocError(std::move(result),
                                executableModelOr.takeError());
    std::move(result).emplace(executableModelOr.takeValue());
  });

  return result;
}

AsyncValueRef<ExecutableModels>
Engine::Context::initModel(AsyncValueRef<CompiledModels> compiledModels,
                           const WeightsRegistryRef &weightsRegistry) const {
  auto result = AsyncValueRef<ExecutableModels>::allocate(cpuDevice);

  // Use andThenAsync to ensure createExecutableModels runs on a cpuDevice
  // worker thread.  The caller may be a foreign thread (e.g. a Python
  // ThreadPoolExecutor thread) that has no AsyncRT cpuDevice set in its
  // thread-local storage; MGP_RT_BufferBorrowV2 (called during Model::setup)
  // requires a valid current cpuDevice.
  std::move(compiledModels)
      .andThenAsync(
          [result = result.copy(), weights = weightsRegistry.copy(),
           this](AsyncValueRef<CompiledModels> &&compiledModelsRef) mutable {
            if (compiledModelsRef.isError())
              return setUnknownLocError(
                  std::move(result),
                  compiledModelsRef.takeDiagnostic().getMessage().copy());

            auto executableModelsOr =
                createExecutableModels(compiledModelsRef.get(), cpuDevice,
                                       getConfig(), weights.getPointer());
            if (executableModelsOr.isError())
              return setUnknownLocError(std::move(result),
                                        executableModelsOr.takeError());

            std::move(result).emplace(executableModelsOr.takeValue());
          });

  return result;
}

//===----------------------------------------------------------------------===//
// Model execution
//===----------------------------------------------------------------------===//

AsyncValueRef<TensorMap>
Engine::Context::execute(AsyncExecutableModel executableModel,
                         AsyncValueRef<TensorMap> inputTensorMap) const {
  auto outputTensorMap = AsyncValueRef<TensorMap>::allocate(cpuDevice);
  AsyncExecutableModel executableModelRef = executableModel.copy();
  executableModel.andThenAsync([this, inTensorMap = inputTensorMap.copy(),
                                outTensorMap = outputTensorMap.copy(),
                                executableModelRef =
                                    std::move(executableModelRef)]() mutable {
    if (executableModelRef.isError()) {
      return std::move(outTensorMap)
          .setToError(executableModelRef.takeDiagnostic());
    }

    AsyncRT::await(inTensorMap);
    auto &inputMap = inTensorMap.get();

    std::unique_ptr<ExecutableModel> &executableModel =
        executableModelRef.get();
    if (!executableModel) {
      return std::move(outTensorMap)
          .setToError({Error::getStaticString("Executable model not available"),
                       UnknownLocationDecoder::getEncodedLocation()});
    }

    ErrorOr<TensorMap> outputsOr =
        executeExecutableModel(*executableModel, inputMap, cpuDevice);
    if (outputsOr.isError()) {
      const Model *model = executableModel->getModel();
      return std::move(outTensorMap)
          .setToError({outputsOr.takeError(),
                       model ? model->getLocation().copy()
                             : UnknownLocationDecoder::getEncodedLocation()});
    }
    std::move(outTensorMap).emplace(outputsOr.takeValue());
  });
  return outputTensorMap;
}

ErrorOr<ArrayRef<QuantTensorSpec>>
Engine::Context::getInputSpecs(AsyncValueRef<CompiledModels> compiledModels,
                               size_t modelIndex) {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    return err.takeError();

  if (auto model = compiledModels->getModel(modelIndex))
    return model->getInputSpecs();

  return Error::getStaticString("model index out of bounds");
}

ErrorOr<ArrayRef<QuantTensorSpec>>
Engine::Context::getOutputSpecs(AsyncValueRef<CompiledModels> compiledModels,
                                size_t modelIndex) {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    return err.takeError();

  if (auto model = compiledModels->getModel(modelIndex))
    return model->getOutputSpecs();

  return Error::getStaticString("model index out of bounds");
}

ErrorOr<ArrayRef<std::string>>
Engine::Context::getInputNames(AsyncValueRef<CompiledModels> compiledModels,
                               size_t modelIndex) {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    return err.takeError();

  if (auto model = compiledModels->getModel(modelIndex))
    return model->getInputNames();

  return Error::getStaticString("model index out of bounds");
}

ErrorOr<ArrayRef<std::string>>
Engine::Context::getOutputNames(AsyncValueRef<CompiledModels> compiledModels,
                                size_t modelIndex) {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    return err.takeError();

  if (auto model = compiledModels->getModel(modelIndex))
    return model->getOutputNames();

  return Error::getStaticString("model index out of bounds");
}

ErrorOr<std::vector<std::string>> Engine::Context::getKernelSummaries(
    const AsyncValueRef<CompiledModels> &compiledModels, size_t modelIndex) {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    return err.takeError();

  if (auto model = compiledModels->getModel(modelIndex))
    return model->getKernelSummaries();

  return Error::getStaticString("model index out of bounds");
}

ErrorOr<Engine::ConfiguredDevice>
Engine::Context::resolveDevice(M::DeviceRef deviceRef) const {
  for (auto &device : getConfig().getDevices())
    if (deviceRef == device.spec->ref)
      return device;
  return Error(Twine("unrecognized device '") + deviceRef.toString() + "'");
}
