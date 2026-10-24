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

#ifndef SDK_ENGINEAPI_PYTHON_INFERENCESESSION_H
#define SDK_ENGINEAPI_PYTHON_INFERENCESESSION_H

#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "GraphCompiler/Support/ModelTensorSpec.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "_core/DLPack/Utils.h"
#include "_core/Driver/Buffer.h"
#include "_core/Driver/Device.h"
#include "max/max_license/API/c/CompiledModels.h"
#include "max/max_license/API/c/EngineConfig.h"
#include "max/max_license/API/c/EngineContext.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/intrusive/ref.h"
#include "nanobind/nanobind.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace M::Core {
class ModelHandle;

/// `Model.execute` failed, often in one of the model's kernels. Raised in
/// Python as `max.engine.ModelExecutionError`.
class ModelExecutionError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

inline llvm::SmallVector<ModelTensorSpec>
getModelTensorSpecs(llvm::ArrayRef<std::string> names,
                    llvm::ArrayRef<QuantTensorSpec> specs) {
  return llvm::map_to_vector(llvm::zip_equal(names, specs),
                             [](const auto &pair) {
                               const auto &[name, spec] = pair;
                               return ModelTensorSpec{name, spec};
                             });
}

class InferenceSession final : public nb::intrusive_base {
public:
  InferenceSession(std::vector<Driver::Python::PyDeviceTy> &&devices,
                   std::vector<std::filesystem::path> &&customExtensions,
                   size_t numThreads);

  AsyncRT::AsyncValueRef<CompiledModels>
  compile(std::filesystem::path modelPath,
          std::vector<std::filesystem::path> &&customExtensions);

  AsyncRT::AsyncValueRef<CompiledModels>
  compile(const nb::capsule &model,
          std::vector<std::filesystem::path> &&customExtensions,
          std::string &&pipelineName, bool tileBasedFusion);

  /// Initialize a single compiled model with the given weights, blocking until
  /// the model is ready. The \p model pointer must remain valid for the
  /// lifetime of the returned handle (i.e. the owning CompiledModels must
  /// outlive it).
  M::AsyncExecutableModel load(const M::Model *model,
                               WeightsRegistryRef weightsRegistry);

  /// Initializes all models in the compiled artifact and returns one
  /// ModelHandle per model in MEF order.
  ///
  /// Args:
  ///   compiledModels: The async compiled models to initialize.
  ///   weightsDict: A dictionary of weights used to initialize the models.
  ///
  /// Returns:
  ///   One ModelHandle per model encoded in the MEF, in MEF order.
  std::vector<ModelHandle> loadAll(AsyncValueRef<CompiledModels> compiledModels,
                                   const nb::dict &weightsDict);

  M::Engine::Context &getRuntimeContext() { return *runtimeContext; }

  /// Gets a list of devices that was initialized on the InferenceSession
  std::vector<Driver::Python::PyDeviceTy> getDevices() const { return devices; }

private:
  void
  addCustomExtensionToCompileConfig(CompilationRequest &compileConfig,
                                    std::filesystem::path custom_extension);
  CompilationRequest
  populateCompileConfig(std::vector<std::filesystem::path> &&customExtensions);
  CompilationRequest sessionCompileConfig;
  M::Engine::Config runtimeConfig;
  std::unique_ptr<M::Engine::Context> runtimeContext;
  std::vector<Driver::Python::PyDeviceTy> devices;
};

/// A ModelHandle holds a reference to the internal (async) compiled model and
/// also a reference to the InferenceSession that was used to create the
/// ModelHandle. This is necessary because the ModelHandle delegates actual
/// loading to the InferenceSession so it is important for it to stay alive as
/// long as a ModelHandle is alive. Also, there can be multiple ModelHandle
/// instances per InferenceSession so a shared_ptr is appropriate. Refer to
/// this issue for a discussion on shared_ptr vs RCRef:
/// https://github.com/modularml/modular/issues/7706

/// We delegate loading to InferenceSession because it is the natural place to
/// store the runtimeContext which is necessary for model loading.
class ModelHandle {
public:
  /// Get a vector of all input tensors' specifications i.e. shape, dtype and
  /// name. Throws a std::invalid_argument if compiled models await fails.
  llvm::SmallVector<ModelTensorSpec> getInputMetadata();

  /// Get a vector of all output tensors' specifications i.e. shape and dtype.
  /// Throws a std::invalid_argument if compiled models await fails.
  llvm::SmallVector<ModelTensorSpec> getOutputMetadata();

  /// Get a vector of all devices setup by the InferenceSession attached to
  /// Model.
  std::vector<Driver::Python::PyDeviceTy> getDevices() const;

  /// Returns the async compiled models held by this handle, for passing to
  /// InferenceSession::loadAll.
  const AsyncValueRef<CompiledModels> &getCompiledModels() const {
    return compiledModels;
  }

  /// Get the dSummary strings from all mgp.generic.execute primitives in the
  /// compiled model's execute function. Each summary describes the fused kernel
  /// composition (e.g. "Epilogue(custom__kv_rope, custom__kv_cache_store)").
  std::vector<std::string> getKernelSummaries();

  /// Get the symbol name of this model (the `sym_name` of its `mo.graph` op,
  /// preserved through MEF serialization). Used by the Python binding's
  /// `Model.name` property so callers of `InferenceSession.load_all` can key
  /// the returned dict by graph name instead of MEF position.
  std::string getName();

  /// Exports the compiled model as a mef to the filename passed.
  /// Args:
  ///   path: Name of the file where the mef is exported.
  void export_mef(const char *path);

  /// Reload weights into this compiled model in-place.
  ///
  /// Reuses the compiled graph and replaces the weight buffers in-place.
  ///
  /// Args:
  ///   weightsDict: A dictionary of weights used to reinitialize the model.
  void reload(const nb::dict &weightsDict);

  /// Drops the host-side weight references held by this handle.
  ///
  /// Releases the weights registry and the owning DLPack references, so the
  /// host weight memory can be freed once the caller drops its own
  /// references. Safe only when every weight was copied to its execution
  /// device during init: reading a host weight after this call is undefined
  /// behavior. `reload` remains usable afterwards.
  void releaseWeights();

  /// Runs inference with the boxed model.
  ///
  /// Args:
  ///   inputTensors: Inputs to be passed to the model.
  ///
  /// Returns:
  ///   List of outputs from the model
  std::vector<Driver::Python::Buffer>
  executeDeviceTensors(std::vector<Driver::Python::Buffer *> &inputTensors);

  /// Capture execution into a device graph collection using per-device keys.
  std::vector<Driver::Python::Buffer>
  capture(const std::vector<uint64_t> &graphKeys,
          std::vector<Driver::Python::Buffer *> &inputs);

  /// Replay the captured device graph collection for per-device keys.
  void replay(const std::vector<uint64_t> &graphKeys,
              std::vector<Driver::Python::Buffer *> &inputs);

  /// Execute eagerly and verify launch trace for per-device keys.
  void debugVerifyReplay(const std::vector<uint64_t> &graphKeys,
                         std::vector<Driver::Python::Buffer *> &inputs);

  /// Await all pending device graph instantiations.
  void awaitDeviceGraphs();

  /// Release captured device graphs for per-device keys, freeing the
  /// underlying device-side graph + working memory once the last reference
  /// drops. Releasing an unknown key is a no-op.
  void releaseCapturedGraphs(const std::vector<uint64_t> &graphKeys);

  // Devices of the model's input tensors.
  std::vector<Driver::Python::PyDeviceTy> inputDevicesPy;

  // Devices of the model's output tensors.
  std::vector<Driver::Python::PyDeviceTy> outputDevicesPy;

private:
  friend class InferenceSession;

  /// Construct a post-load handle for a single model within a compiled
  /// artifact. Used by InferenceSession::loadAll() to produce per-model
  /// handles.
  ModelHandle(AsyncRT::AsyncValueRef<CompiledModels> compiledModels,
              nb::ref<InferenceSession> sessionPtr,
              M::AsyncExecutableModel loadedModel,
              WeightsRegistryRef weightsRegistry,
              std::shared_ptr<SmallVector<DLPack::DLManagedTensorWrapper>>
                  managedTensors,
              size_t modelIndex);

  /// Get a vector of all input tensors' device.
  /// Throws a std::invalid_argument if getting the device for any particular
  /// input is unsuccessful.
  std::vector<Driver::Python::PyDeviceTy> getInputDevices() const;

  /// Get a vector of all output tensors' device.
  /// Throws a std::invalid_argument if getting the device for any particular
  /// output is unsuccessful.
  std::vector<Driver::Python::PyDeviceTy> getOutputDevices() const;

  /// Internal API to allow for the execution of a model without tying it to how
  /// the inputs are prepared.
  std::vector<Driver::Python::Buffer>
  executePreparedDeviceTensors(M::Engine::Context &runtimeCtx,
                               SmallVectorImpl<AnyAsyncValueRef> &inputValues);

  /// Return the cpuDevice context of the associated InferenceSession.
  M::Engine::Context &getRuntimeContext() {
    return inferenceSession->getRuntimeContext();
  }

  // NOTE: inferenceSession must be declared before the models so that
  // inferenceSession is destroyed last.
  // This is because inferenceSession owns the AsyncRT::CPUDevice that the
  // models rely on to deallocate their buffers.
  nb::ref<InferenceSession> inferenceSession;
  AsyncValueRef<CompiledModels> compiledModels;
  M::AsyncExecutableModel loadedModel;
  // All handles from the same loadAll() call share this vector so that the
  // DLPack wrappers (and the numpy/torch arrays they pin) stay alive as long
  // as *any* handle from the batch is alive.  The weights registry holds raw
  // pointers into those arrays, so it must come after managedTensors in the
  // declaration (destroyed first) to keep the ordering safe.
  std::shared_ptr<SmallVector<DLPack::DLManagedTensorWrapper>> managedTensors;
  WeightsRegistryRef weightsRegistry;
  /// Index of this model within the compiled artifact's ModelList.
  size_t modelIndex;
};

} // namespace M::Core

#endif // SDK_ENGINEAPI_PYTHON_INFERENCESESSION_H
