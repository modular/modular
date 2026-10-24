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

#include "_core/Engine/InferenceSession.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "Init/DevelopmentSignalHandler.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "Support/Filesystem/Paths.h"
#include "Support/Profiling/Ranges.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "_core/DLPack/Utils.h"
#include "_core/Driver/Buffer.h"
#include "_core/Engine/DebugConfig.h"
#include "max/max_license/API/c/CompiledModels.h"
#include "max/max_license/API/c/EngineContext.h"
#include "max/max_license/API/c/ModelExecution.h"
#include "max/max_license/API/c/Telemetry.h"
#include "mlir/CAPI/IR.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/filesystem.h" // IWYU pragma: keep (type caster)
#include "nanobind/stl/shared_ptr.h" // IWYU pragma: keep (type caster)
#include "nanobind/stl/string.h"     // IWYU pragma: keep (type caster)
#include "nanobind/stl/vector.h"     // IWYU pragma: keep (type caster)
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/bit.h"
#include <Python.h>
#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace nb = nanobind;

using namespace M;
using namespace M::Core;

using DeviceToPyDeviceMap =
    llvm::SmallDenseMap<M::DeviceRef, Driver::Python::PyDeviceTy>;

/// Convert model execution results to Python buffers.
static std::vector<Driver::Python::Buffer>
extractPyBufferResults(ArrayRef<M::GraphCompiler::Driver::ExecResult> results,
                       ArrayRef<Driver::Python::PyDeviceTy> devices) {
  // Reconstruct a set of output tensor objects to be returned to the user.
  std::vector<Driver::Python::Buffer> pyResults;
  pyResults.reserve(results.size());

  // Aliasing outputs share one device buffer, so they must share one storage
  // for the Python objects to alias the way they did before.
  llvm::SmallDenseMap<M::Driver::DeviceBuffer *,
                      std::shared_ptr<Driver::Python::BufferStorage>>
      storageCache;
  size_t index = 0;
  for (const auto &result : results) {
    auto [it, inserted] =
        storageCache.try_emplace(result.buffer.getPointer(), nullptr);
    if (inserted)
      it->second = std::make_shared<Driver::Python::BufferStorage>(
          result.buffer, result.spec);
    pyResults.emplace_back(it->second, devices[index], nullptr);
    ++index;
  }

  return pyResults;
}

/// Get a vector of all input tensors' specifications i.e. shape, dtype and
/// name. Throws a std::invalid_argument if compiled models await fails.
SmallVector<ModelTensorSpec> M::Core::ModelHandle::getInputMetadata() {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    throw std::invalid_argument(err.getError());

  const M::Model *model = compiledModels.get().getModel(modelIndex);
  return getModelTensorSpecs(model->getInputNames(), model->getInputSpecs());
}

/// Resolves device references against session devices.
/// Creates CPU devices if needed, throws for missing accelerator devices.
static std::vector<Driver::Python::PyDeviceTy>
resolveDeviceRefs(const ArrayRef<M::DeviceRef> deviceRefs,
                  const DeviceToPyDeviceMap &sessionDeviceMap,
                  StringRef contextString) {
  std::vector<Driver::Python::PyDeviceTy> res;
  res.reserve(deviceRefs.size());

  for (const auto &[idx, deviceRef] : llvm::enumerate(deviceRefs)) {
    auto it = sessionDeviceMap.find(deviceRef);
    if (it != sessionDeviceMap.end()) {
      // Device found in session map.
      res.emplace_back(it->second);
      continue;
    }

    // Device not found in session map.
    if (deviceRef.label == kCPULabel) {
      // Handle CPU specially: create/get if not present.
      // TODO(GEX-2092): always create CPU device on the session.
      res.emplace_back(Driver::Python::getCpuDevice(deviceRef.id));
    } else {
      // Accelerator device not found, throw error.
      throw std::invalid_argument("Loaded Model " + contextString.str() +
                                  " idx=" + std::to_string(idx) +
                                  " uses device=" + deviceRef.toString() +
                                  " which was not set up in InferenceSession");
    }
  }
  return res;
}

/// Get a vector of all input tensors' device.
/// Throws a std::invalid_argument if getting the device for any particular
/// input is unsuccessful.
std::vector<Driver::Python::PyDeviceTy>
M::Core::ModelHandle::getInputDevices() const {
  if (!loadedModel)
    return {};

  DeviceToPyDeviceMap deviceMap;
  for (const auto &pyDevice : inferenceSession->getDevices())
    deviceMap.insert({pyDevice->deviceSpec.ref, pyDevice});

  auto *executableModel = loadedModel->get();
  if (!executableModel)
    return {};
  auto deviceRefs = executableModel->getModel()->getInputDevices();
  return resolveDeviceRefs(deviceRefs, deviceMap, "input");
}

/// Get a vector of all output tensors' specifications i.e. shape and dtype.
/// Throws a std::invalid_argument if compiled models await fails.
SmallVector<ModelTensorSpec> M::Core::ModelHandle::getOutputMetadata() {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    throw std::invalid_argument(err.getError());

  const M::Model *model = compiledModels.get().getModel(modelIndex);
  return getModelTensorSpecs(model->getOutputNames(), model->getOutputSpecs());
}

std::vector<std::string> M::Core::ModelHandle::getKernelSummaries() {
  auto summariesOr =
      Engine::Context::getKernelSummaries(compiledModels, modelIndex);
  if (summariesOr.isError())
    throw std::invalid_argument(summariesOr.getError());
  return summariesOr.takeValue();
}

std::string M::Core::ModelHandle::getName() {
  if (auto err = AsyncRT::awaitOrError(compiledModels))
    throw std::invalid_argument(err.getError());

  const M::Model *model = compiledModels.get().getModel(modelIndex);
  return model->getName().str();
}

/// Get a vector of all output tensors' device.
/// Throws a std::invalid_argument if getting the device for any particular
/// output is unsuccessful.
std::vector<Driver::Python::PyDeviceTy>
M::Core::ModelHandle::getOutputDevices() const {
  if (!loadedModel)
    return {};

  DeviceToPyDeviceMap deviceMap;
  for (const auto &pyDevice : inferenceSession->getDevices())
    deviceMap.insert({pyDevice->deviceSpec.ref, pyDevice});

  auto *executableModel = loadedModel->get();
  if (!executableModel)
    return {};
  auto deviceRefs = executableModel->getModel()->getOutputDevices();
  return resolveDeviceRefs(deviceRefs, deviceMap, "output");
}

/// Get a vector of all devices setup by the InferenceSession
std::vector<Driver::Python::PyDeviceTy>
M::Core::ModelHandle::getDevices() const {
  return inferenceSession->getDevices();
}

/// Private constructor for post-load per-model handles produced by loadAll().
M::Core::ModelHandle::ModelHandle(
    AsyncRT::AsyncValueRef<CompiledModels> compiledModels,
    nb::ref<InferenceSession> sessionPtr,
    M::AsyncExecutableModel loadedModelArg, WeightsRegistryRef weightsReg,
    std::shared_ptr<SmallVector<DLPack::DLManagedTensorWrapper>>
        managedTensorsArg,
    size_t index)
    : inferenceSession{std::move(sessionPtr)},
      compiledModels{std::move(compiledModels)},
      loadedModel{std::move(loadedModelArg)},
      managedTensors{std::move(managedTensorsArg)},
      weightsRegistry{std::move(weightsReg)}, modelIndex{index} {}

/// Build a WeightsRegistry and shared managed-tensor vector from a Python dict.
///
/// Handles both read-only numpy arrays (borrowed pointer) and objects
/// implementing the DLPack protocol (shared ownership via DLManagedTensor).
/// Both the registry and the managed-tensor vector must outlive any model
/// that uses the registry.
static std::pair<WeightsRegistryRef,
                 std::shared_ptr<SmallVector<DLPack::DLManagedTensorWrapper>>>
buildWeightsFromDict(const nb::dict &weightsDict) {
  size_t dictSize = weightsDict.size();
  std::vector<const char *> weightNames;
  weightNames.reserve(dictSize);
  std::vector<const void *> weightPtrs;
  weightPtrs.reserve(dictSize);
  SmallVector<DLPack::DLManagedTensorWrapper> localManagedTensors;
  localManagedTensors.reserve(dictSize);

  for (const auto &[name, array] : weightsDict) {
    weightNames.push_back(nb::borrow<nb::str>(name).c_str());

    // TODO(MSDK-976): Older versions of numpy don't support the dlpack protocol
    // for read-only arrays (e.g. that come from memmapped weights). So for now
    // we just borrow their pointers and assume (hope, really) that they will
    // outlive our model execution.
    if (nb::isinstance<nb::ndarray<nb::ro>>(array)) {
      // We're specifically testing for read-only arrays, but a read-write array
      // also satisfies the above check. So make sure it isn't a read-write.
      if (!nb::isinstance<nb::ndarray<>>(array)) {
        auto npArray = nb::cast<nb::ndarray<nb::ro>>(array);
        weightPtrs.push_back(npArray.data());
        continue;
      }
    } else if (!nb::hasattr(array, "__dlpack__") ||
               !nb::hasattr(array, "__dlpack_device__")) {
      throw std::invalid_argument(
          "weight registry element does not implement dlpack");
    }

    // Take a shared owning reference to the underlying data from the input
    // weightsDict. Using the dlpack API ensures that the producer framework
    // (e.g. numpy) gets notified, and keeps the underlying array alive as long
    // as we need it.
    auto arrayOrTensor = nb::borrow<nb::object>(array);

    // This also ensures that the device is supported by dlpack.
    auto [_, streamID] = Driver::Python::getPyDevice(arrayOrTensor);

    DLPack::DLManagedTensorWrapper &managed = localManagedTensors.emplace_back(
        DLPack::callDLPack(arrayOrTensor, streamID));
    weightPtrs.push_back(managed->dl_tensor.data);
  }

  auto weightsRegistryOr = WeightsRegistry::create(
      {weightNames.data(), weightNames.size()},
      {llvm::bit_cast<const std::byte **>(weightPtrs.data()),
       weightPtrs.size()});
  if (weightsRegistryOr.isError())
    throw std::invalid_argument(weightsRegistryOr.getError());

  // Wrap the managed tensors in a shared_ptr so callers can share ownership
  // across multiple ModelHandles (keeping DLPack wrappers alive until all
  // consumers are destroyed).
  return {weightsRegistryOr.takeValue(),
          std::make_shared<SmallVector<DLPack::DLManagedTensorWrapper>>(
              std::move(localManagedTensors))};
}

M::AsyncExecutableModel
M::Core::InferenceSession::load(const M::Model *model,
                                WeightsRegistryRef weightsRegistry) {
  auto execModelAsync = runtimeContext->initModel(model, weightsRegistry);
  // Release the GIL during model initialization to allow other Python threads
  // to run (e.g., for logging progress messages during long loads).
  nb::gil_scoped_release release;
  if (auto err = Engine::Context::await(execModelAsync.copy()))
    throw std::invalid_argument(err.takeError().get());
  return execModelAsync;
}

/// Initializes all models in the compiled artifact and returns one ModelHandle
/// per model in MEF order.
std::vector<M::Core::ModelHandle>
M::Core::InferenceSession::loadAll(AsyncValueRef<CompiledModels> compiledModels,
                                   const nb::dict &weightsDict) {
  auto [sharedWeightsRegistry, sharedManagedTensors] =
      buildWeightsFromDict(weightsDict);

  // Ensure the compiled models are ready before accessing individual Model*s.
  // Release the GIL during the await to allow other Python threads to run
  // (e.g., for logging progress messages during long compilations).
  {
    nb::gil_scoped_release release;
    if (auto err = Engine::Context::await(compiledModels.copy()))
      throw std::invalid_argument(err.takeError().get());
  }

  const CompiledModels &compiled = compiledModels.get();
  size_t numModels = compiled.getNumModels();

  std::vector<ModelHandle> results;
  results.reserve(numModels);
  for (auto &&[index, model] : llvm::enumerate(compiled)) {
    M::AsyncExecutableModel execModel =
        load(model.get(), sharedWeightsRegistry);
    ModelHandle handle(compiledModels.copy(), nb::ref<InferenceSession>(this),
                       std::move(execModel), sharedWeightsRegistry,
                       sharedManagedTensors, index);
    // Build input and output device vectors exactly once at model load time.
    handle.inputDevicesPy = handle.getInputDevices();
    handle.outputDevicesPy = handle.getOutputDevices();
    results.push_back(std::move(handle));
  }
  return results;
}

void M::Core::ModelHandle::reload(const nb::dict &weightsDict) {
  // Drop current resource references before attempting to reload.
  weightsRegistry = {};
  managedTensors = {};
  loadedModel = {};

  auto [newRegistry, newManagedTensors] = buildWeightsFromDict(weightsDict);
  loadedModel = inferenceSession->load(
      compiledModels.get().getModel(modelIndex), newRegistry);
  weightsRegistry = std::move(newRegistry);
  managedTensors = std::move(newManagedTensors);
  inputDevicesPy = getInputDevices();
  outputDevicesPy = getOutputDevices();
}

void M::Core::ModelHandle::releaseWeights() {
  weightsRegistry = {};
  managedTensors = {};
}

void M::Core::ModelHandle::export_mef(const char *path) {
  if (!compiledModels || !compiledModels->getBuffer()) {
    throw std::invalid_argument(
        "model is null or compiled model isn't available");
  }
  // TODO MSDK-1026: Delete the `remove` call below.
  // Remove the file if it exists. Currently, the getFile maps the file to be
  // written to. But if the new content is shorter than the original file
  // content, there will be leftover data from the old version. A better fix is
  // to provide a createFile in WriteableBuffer.
  (void)llvm::sys::fs::remove(path);
  auto writeOr = WriteableBuffer::getFile(
      path, compiledModels->getBuffer()->getBufferSize(), 0);
  if (writeOr.isError()) {
    throw std::invalid_argument(writeOr.getError());
  }
  // Handle error
  WriteableBufferRef os = writeOr.takeValue();
  *os << compiledModels->getBuffer()->getBuffer();
}

/// Verifies that the input tensor's device matches that expected by the model.
/// It is only OK to pass a device pinned host tensor when a CPU graph input is
/// expected. It is not allowed to pass a pinned tensor in place of a GPU input.
/// Raises an `IndexError` if the tensor input index is out of bounds for the
/// model.
/// Raises a `TypeError` if the input tensor has a different device than that
/// expected by the model.
static void
verifyInputDevice(const Driver::Python::Buffer *tensor,
                  ArrayRef<Driver::Python::PyDeviceTy> modelInputDevices,
                  size_t tensorInputIdx) {
  if (tensorInputIdx >= modelInputDevices.size()) {
    std::string errorMsg = ("tensor input index " + Twine(tensorInputIdx) +
                            " out of bounds for model with " +
                            Twine(modelInputDevices.size()) + " devices")
                               .str();
    throw nb::index_error(errorMsg.c_str());
  }

  Driver::Python::PyDeviceTy modelInputDevice =
      modelInputDevices[tensorInputIdx];

  // A pinned tensor can only be used in place of a CPU graph input.
  // It cannot be used in place of a GPU graph input.
  if (tensor->isPinned()) {
    if (modelInputDevice->isHost()) {
      return;
    } else {
      std::string errorMsg =
          (Twine("expected argument ") + Twine(tensorInputIdx) +
           " to be a device tensor on " + modelInputDevice->getAsString() +
           " but found pinned tensor on " + tensor->getDevice()->getAsString() +
           " instead. Pinned tensors can only be used in place of CPU graph "
           "inputs.")
              .str();
      throw nb::type_error(errorMsg.c_str());
    }
  }
  // Otherwise, the tensor must be on matching devices.
  if (*tensor->getDevice() == *modelInputDevice)
    return;

  // Mismatch! Throw an error.
  std::string errorMsg =
      (Twine("expected argument ") + Twine(tensorInputIdx) +
       " to be on device " + modelInputDevice->getAsString() +
       ", but was on device " + tensor->getDevice()->getAsString())
          .str();
  throw nb::type_error(errorMsg.c_str());
}

/// Validates an ordered list of optional TensorSpecs for their compatibility
/// with a provided model. This is used when our execution inputs are an ordered
/// list of tensors or types that we don't verify.
static void validateInputSpecList(
    const AsyncValueRef<CompiledModels> &models, size_t modelIndex,
    const std::vector<Driver::Python::Buffer *> &inputTensors) {
  ErrorOr<ArrayRef<QuantTensorSpec>> inputSpecsOr =
      Engine::Context::getInputSpecs(models.copy(), modelIndex);
  if (inputSpecsOr.isError())
    throw std::invalid_argument(inputSpecsOr.takeError().get());

  ArrayRef<QuantTensorSpec> inputSpecs = inputSpecsOr.get();

  // We need to validate that the user has provided the correct number of
  // inputs.
  size_t numInputs = inputTensors.size();
  if (numInputs != inputSpecs.size()) {
    throw std::invalid_argument(
        "Number of inputs (" + std::to_string(numInputs) +
        ") does not match expected number (" +
        std::to_string(inputSpecs.size()) + ") for model");
  }

  for (size_t i = 0; i < numInputs; ++i) {
    const TensorSpec &providedSpec = inputTensors[i]->getSpec();
    const TensorSpec &expectedSpec = inputSpecs[i];

    // Use shared validation logic
    ErrorOrSuccess validationResult =
        M::GraphCompiler::Driver::validateInputSpec(providedSpec, expectedSpec);
    if (validationResult.isError())
      throw std::invalid_argument("Input at position " + std::to_string(i) +
                                  ": " + validationResult.takeError().get());
  }
}

// Entrypoint for executing Python DeviceTensors.
std::vector<Driver::Python::Buffer> M::Core::ModelHandle::executeDeviceTensors(
    std::vector<Driver::Python::Buffer *> &inputTensors) {
  TRACY_ZONE_SCOPED_NC("ModelHandle::executeDeviceTensors", TRACY_COLOR_YELLOW);

  if (!loadedModel)
    return {};

  // We now need to validate if the tensors the user provided are of the correct
  // shape and dtype.
  validateInputSpecList(compiledModels, modelIndex, inputTensors);

  M::Engine::Context &runtimeCtx = getRuntimeContext();

  const size_t numInputs = inputTensors.size();
  // Extract the objects backing the inputs and feed them into the execution
  // method.
  SmallVector<AsyncRT::AnyAsyncValueRef> valuePtrs;
  valuePtrs.reserve(numInputs);

  // Track the tensor input indices separately to index into the input devices.
  // This is because only tensor input devices are encoded in the `Model`.
  size_t tensorInputIdx = 0;
  for (auto &tensor : inputTensors) {
    // Check that the tensor input device is valid.
    verifyInputDevice(tensor, inputDevicesPy, tensorInputIdx);

    ErrorOr<std::shared_ptr<Driver::Python::BufferStorage>> storageOr =
        tensor->getInputStorage();
    if (storageOr.isError())
      throw std::runtime_error(storageOr.takeError().get());

    // Marshal the tensor into an AsyncValue for model execution.
    valuePtrs.push_back(M::GraphCompiler::Driver::prepareInput(
        runtimeCtx.getCPUDevice(), storageOr.takeValue()->buf,
        tensor->getSpec()));

    ++tensorInputIdx;
  }

  // Each user-visible model.execute() that reaches the executor counts as
  // one profiler step — whether the executor returns cleanly or throws. This
  // is the production step boundary: the count drives the warmup/active
  // step-window state machine (max-debug.profiling-warmup-steps /
  // profiling-active-steps) and forwards a libkineto iteration marker while
  // a trace is live. executePreparedDeviceTensors can dispatch kernels
  // before failing on a driver error, so the step boundary needs to fire
  // even on the throw path or libkineto's correlation-ID stream would lose
  // its iteration markers.
  // The disabled path is a single predicted-branch isEnabled() check and
  // early return inside M::Profiling::step() — the same fast-path contract
  // as rangeBegin/rangeEnd/RangeScope, and well below the noise floor of
  // the executor body. Perf gates (#86467) validate end-to-end.
  struct ScopedStep {
    ~ScopedStep() {
      // This dtor fires during stack unwinding when the executor throws.
      // step() can call into libkineto on the active path; a throw escaping
      // here would cross the (implicitly noexcept) destructor boundary
      // mid-unwind and call std::terminate. A profiler hiccup must never
      // turn an executor error into a hard abort, so swallow.
      try {
        M::Profiling::step();
      } catch (...) {
      }
    }
  } stepGuard;

  return executePreparedDeviceTensors(runtimeCtx, valuePtrs);
}

std::vector<Driver::Python::Buffer> ModelHandle::executePreparedDeviceTensors(
    M::Engine::Context &runtimeCtx,
    SmallVectorImpl<AnyAsyncValueRef> &inputValues) {
  auto execResults = [&]() -> GraphCompiler::Driver::ExecResultType {
    // We need to release the GIL while we execute. This is to prevent
    // situations where other threads are attempting to decref DLPack tensors
    // that are being used by our engine and the `execute` call holds the GIL.
    // This situation can cause a deadlock.
    nb::gil_scoped_release release;
    auto result = M::GraphCompiler::Driver::executeFromDriver(
        runtimeCtx, loadedModel.copy(), inputValues);
    if (result.isError())
      throw ModelExecutionError(result.takeError().get());

    return result.takeValue();
  }();

  return extractPyBufferResults(execResults, outputDevicesPy);
}

M::Core::InferenceSession::InferenceSession(
    std::vector<Driver::Python::PyDeviceTy> &&devices,
    std::vector<std::filesystem::path> &&customExtensions, size_t numThreads) {
  // Ideally this does direct initialization, populateCompileConfig reads
  // the current value.
  sessionCompileConfig = populateCompileConfig(std::move(customExtensions));

  // Defaults to 0 (use all threads)
  runtimeConfig.setNumThreads(numThreads);

  this->devices = std::move(devices);
  for (const auto &device : this->devices)
    runtimeConfig.addDevice(
        {std::make_shared<const M::DeviceSpec>(device->deviceSpec),
         device->ctx});

  runtimeConfig.setAPILanguage("python");

  auto runtimeCtxOr = Engine::Context::create(runtimeConfig);
  if (runtimeCtxOr.isError())
    throw std::invalid_argument(runtimeCtxOr.takeError().get());

  runtimeContext =
      std::make_unique<M::Engine::Context>(runtimeCtxOr.takeValue());

  // Enable Python stack traces for signal handler
  // Actual Python stack traces are handled by faulthandler for async signal
  // safety
  M::Init::enablePythonStackTraceCallback();

  // Auto-start the profiler when requested via configuration
  // (max-debug.profiling-enabled, settable through the
  // MODULAR_MAX_DEBUG_PROFILING_ENABLED / MODULAR_DEBUG env vars,
  // modular.cfg, or session.debug). Session construction is the earliest
  // common point on every entry path — before model load and therefore
  // before MLRT retains a device's primary context — so a capture spans
  // compilation too. On AMD the same knob makes the profiler host glue load
  // and register the plugin at device init; this enable() then records via
  // the deferred activation (see activatePendingTrace() in
  // Support/Profiling/Ranges.h). enable() is idempotent. The trace is flushed
  // by the profiler plugin's own exit hook, so no Python involvement is
  // required end to end.
  if (DebugConfig::instance().getProfilingEnabled()) {
    M::Profiling::enable();
    if (!M::Profiling::haveProfiler()) {
      // Once per process: several sessions would otherwise repeat it.
      static std::once_flag pluginMissingWarned;
      std::call_once(pluginMissingWarned, [] {
        std::fprintf(stderr,
                     "info: profiling was requested (max-debug.profiling-"
                     "enabled) but the profiler plugin (libMAXProfilerPlugin"
                     ".so) was not found; nothing will be recorded. Deploy the "
                     "plugin next to the MAX libraries or set "
                     "MODULAR_PROFILER_PLUGIN to its path.\n");
      });
    }
  }
}

void M::Core::InferenceSession::addCustomExtensionToCompileConfig(
    CompilationRequest &config, std::filesystem::path path) {
  if (M::Filesystem::isMojoBinaryPackagePath(path)) {
    // Mojo extension.
    if (std::error_code ec; !std::filesystem::exists(path, ec) || ec)
      throw std::invalid_argument("invalid path to kernel library: " +
                                  path.string());

    config.addKernelLibraryPath(std::move(path));
  } else {
    throw std::invalid_argument("Error: unknown custom extension type of `" +
                                path.string() + "`.");
  }
}

CompilationRequest M::Core::InferenceSession::populateCompileConfig(
    std::vector<std::filesystem::path> &&customExtensions) {
  // Clone the existing compile config
  auto compileConfig = sessionCompileConfig;

  for (auto path : customExtensions)
    addCustomExtensionToCompileConfig(compileConfig, std::move(path));

  return compileConfig;
}

static AsyncValueRef<CompiledModels>
compileModel(const M::Engine::Context &runtimeContext,
             CompilationRequest &&config) {
  AsyncValueRef<CompiledModels> compiledModels =
      runtimeContext.compileModel(std::move(config));
  if (compiledModels.isError()) {
    auto &telemetryCtx =
        *runtimeContext.maxContext->get<GML::MaxEngineTelemetryContext>();
    Error error = compiledModels.getDiagnostic().getMessage().copy();
    Telemetry::emitCompileInfo("compile.failure", telemetryCtx,
                               std::optional{error.copy()});

    throw std::invalid_argument(error.get());
  }
  return compiledModels;
}

AsyncRT::AsyncValueRef<CompiledModels> M::Core::InferenceSession::compile(
    std::filesystem::path modelPath,
    std::vector<std::filesystem::path> &&customExtensions) {
  CompilationRequest compileConfig =
      populateCompileConfig(std::move(customExtensions));
  compileConfig.setModelPath(std::move(modelPath));
  // Compile async, so it's safe to hold the gil here
  auto compiledModels{compileModel(*runtimeContext, std::move(compileConfig))};
  return compiledModels;
}

AsyncRT::AsyncValueRef<CompiledModels> M::Core::InferenceSession::compile(
    const nb::capsule &model,
    std::vector<std::filesystem::path> &&customExtensions,
    std::string &&pipelineName, bool tileBasedFusion) {
  CompilationRequest compileConfig =
      populateCompileConfig(std::move(customExtensions));
  // Set pipeline name. Tracked for telemetry.
  compileConfig.setPipelineName(std::move(pipelineName));
  // Opt into the tile-based programming model if requested.
  compileConfig.setTileBasedFusion(tileBasedFusion);
  // When the format is Max Graph, model is a opaque pointer of graph.
  // So we cast it to a pointer representation and pass it along.
  // The C API will open appropriate dylib for the format and pass the
  // pointer to the library. We have to make sure the pointer is passed down
  // to the library which created the pointer. This is to ensure mlir type
  // id invariants are not broken.
  compileConfig.setModelSource(ModelSource{
      .source = model.data(),
      .fileFormat = frameworkToFileFormat(),
  });
  // Compile async, so it's safe to hold the gil here
  AsyncValueRef<CompiledModels> asyncModel =
      compileModel(*runtimeContext, std::move(compileConfig));
  return asyncModel;
}

// ===== ModelHandle Graph Capture/Replay Methods ===== //

static SmallVector<ModelInputBuffer>
collectInputBuffers(const AsyncValueRef<CompiledModels> &compiledModels,
                    size_t modelIndex,
                    const std::vector<Driver::Python::Buffer *> &inputs,
                    ArrayRef<Driver::Python::PyDeviceTy> inputDevices) {
  validateInputSpecList(compiledModels, modelIndex, inputs);

  SmallVector<ModelInputBuffer> buffers;
  buffers.reserve(inputs.size());

  size_t tensorInputIdx = 0;
  for (auto *tensor : inputs) {
    verifyInputDevice(tensor, inputDevices, tensorInputIdx);
    ErrorOr<std::shared_ptr<Driver::Python::BufferStorage>> storageOr =
        tensor->getInputStorage();
    if (storageOr.isError())
      throw std::runtime_error(storageOr.takeError().get());
    std::shared_ptr<Driver::Python::BufferStorage> storage =
        storageOr.takeValue();
    buffers.push_back({storage->buf, storage->spec});
    ++tensorInputIdx;
  }

  return buffers;
}

std::vector<Driver::Python::Buffer>
M::Core::ModelHandle::capture(const std::vector<uint64_t> &graphKeys,
                              std::vector<Driver::Python::Buffer *> &inputs) {
  if (!loadedModel)
    throw std::runtime_error("Model not loaded");

  auto *executableModel = loadedModel->get();
  if (!executableModel)
    throw std::runtime_error("Executable model not available");

  auto buffers =
      collectInputBuffers(compiledModels, modelIndex, inputs, inputDevicesPy);
  auto result = [&] {
    nb::gil_scoped_release release;
    return executableModel->capture(graphKeys, buffers);
  }();
  if (result.isError())
    throw std::runtime_error(result.takeError().get());

  M::Engine::Context &runtimeCtx = getRuntimeContext();
  SmallVector<AnyAsyncValueRef> results = result.takeValue();
  auto execResults = GraphCompiler::Driver::extractDriverResults(
      runtimeCtx, loadedModel.copy(), results);

  if (execResults.isError())
    throw std::runtime_error(execResults.takeError().get());

  return extractPyBufferResults(*execResults, outputDevicesPy);
}

void M::Core::ModelHandle::replay(
    const std::vector<uint64_t> &graphKeys,
    std::vector<Driver::Python::Buffer *> &inputs) {
  if (!loadedModel)
    throw std::runtime_error("Model not loaded");

  auto *executableModel = loadedModel->get();
  if (!executableModel)
    throw std::runtime_error("Executable model not available");

  auto buffers =
      collectInputBuffers(compiledModels, modelIndex, inputs, inputDevicesPy);
  auto result = [&] {
    nb::gil_scoped_release release;
    return executableModel->replay(graphKeys, buffers);
  }();
  if (result.isError())
    throw std::runtime_error(result.takeError().get());
}

void M::Core::ModelHandle::debugVerifyReplay(
    const std::vector<uint64_t> &graphKeys,
    std::vector<Driver::Python::Buffer *> &inputs) {
  if (!loadedModel)
    throw std::runtime_error("Model not loaded");

  auto *executableModel = loadedModel->get();
  if (!executableModel)
    throw std::runtime_error("Executable model not available");

  auto buffers =
      collectInputBuffers(compiledModels, modelIndex, inputs, inputDevicesPy);

  // Call ExecutableModel and convert errors to exceptions
  auto result = [&] {
    nb::gil_scoped_release release;
    return executableModel->debugVerifyReplay(graphKeys, buffers);
  }();
  if (result.isError())
    throw std::runtime_error(result.takeError().get());
}

void M::Core::ModelHandle::awaitDeviceGraphs() {
  if (!loadedModel)
    throw std::runtime_error("Model not loaded");

  auto *executableModel = loadedModel->get();
  if (!executableModel)
    throw std::runtime_error("Executable model not available");

  nb::gil_scoped_release release;
  auto result = executableModel->awaitDeviceGraphs();
  if (result.isError())
    throw std::runtime_error(result.takeError().get());
}

void M::Core::ModelHandle::releaseCapturedGraphs(
    const std::vector<uint64_t> &graphKeys) {
  if (!loadedModel)
    throw std::runtime_error("Model not loaded");

  auto *executableModel = loadedModel->get();
  if (!executableModel)
    throw std::runtime_error("Executable model not available");

  auto result = [&] {
    nb::gil_scoped_release release;
    return executableModel->releaseCapturedGraphs(graphKeys);
  }();
  if (result.isError())
    throw std::runtime_error(result.takeError().get());
}
