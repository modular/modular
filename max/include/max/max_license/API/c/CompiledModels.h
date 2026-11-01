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

#ifndef GENERICML_API_COMPILEDMODELS_H
#define GENERICML_API_COMPILEDMODELS_H

#include "GraphCompiler/Driver/Common/DriverConfig.h"
#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "GraphCompiler/FrameworkFrontend.h"
#include "GraphCompiler/MEFModel/MEFModel.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/Buffer.h"
#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/ML/SizeUtils.h"
#include "Support/SymbolExport.h"
#include "max/max_license/API/c/EngineConfig.h"
#include "max/max_license/API/c/SourceModel.h"
#include "max/max_license/API/c/detail/ModelContext.h"
#include <filesystem>
#include <string>

namespace M {

/// Returns the preferred device
///   - the first accelerator in devices
///   - else the first CPU
///   — else nullptr if devices is empty
inline const M::Engine::ConfiguredDevice *
selectDevice(ArrayRef<M::Engine::ConfiguredDevice> devices) {
  for (const auto &device : devices)
    // Any non-CPU label counts as an accelerator
    if (device.spec->ref.label != kCPULabel)
      return &device;
  return devices.empty() ? nullptr : devices.data();
}

/// Returns the driver device config for `selectDevice(devices)`, or nullptr
/// if \p devices is empty.
inline std::unique_ptr<DriverConfig>
selectDeviceConfig(ArrayRef<M::Engine::ConfiguredDevice> devices) {
  const auto *device = selectDevice(devices);
  if (!device)
    return nullptr;
  const DeviceSpec &spec = *device->spec;
  return std::make_unique<DeviceConfig>(spec.ref.label, spec.ref.id, spec);
}

/// The internal type held by a `M_CompileConfig` that is created from
/// `M_newCompileConfig()`.
/// The `CompilationRequest` type represents the configs both for local and
/// CAS-aware compilation.
/// Note that this type can be used for a single compilation call. A new
/// `CompilationRequest` must be used for subsequent compilations.
class CompilationRequest {
public:
  CompilationRequest() = default;

  void addFrameworkConfig(std::unique_ptr<DriverConfig> config) {
    passthroughConfigs.push_back(std::move(config));
  }

  ArrayRef<std::unique_ptr<DriverConfig>> getFrameworkConfigs() const {
    return passthroughConfigs;
  }

  CompilationRequest(const CompilationRequest &other) {
    modelSource = other.modelSource;
    pipelineName = other.pipelineName;
    for (auto &config : other.passthroughConfigs)
      passthroughConfigs.push_back(config->clone());
  }

  CompilationRequest(CompilationRequest &&other) = default;
  CompilationRequest &operator=(CompilationRequest &&other) = default;

  CompilationRequest &operator=(const CompilationRequest &other) {
    modelSource = other.modelSource;
    pipelineName = other.pipelineName;
    for (auto &config : other.passthroughConfigs)
      passthroughConfigs.push_back(config->clone());
    return *this;
  }

  /// Get the underlying source model representation.
  std::optional<ModelSource> getModelSource() const {
    if (const ModelSource *val = std::get_if<ModelSource>(&modelSource))
      return *val;
    return std::nullopt;
  }

  /// Get the MAX Graph pipeline name.
  /// Only included if model source is a MAX Graph.
  std::optional<std::string> getPipelineName() const { return pipelineName; }

  /// Get the underlying path on the filesystem for where a model is.  This is
  /// used when compiling a model if the model ID is not set.
  std::optional<std::filesystem::path> getModelPath() const {
    if (const std::filesystem::path *path =
            std::get_if<std::filesystem::path>(&modelSource))
      return *path;
    return std::nullopt;
  }

  /// Set the underlying source model representation.
  void setModelSource(ModelSource source) {
    // Do not overwrite existing values.
    assert(std::holds_alternative<std::monostate>(modelSource));
    modelSource = source;
  }

  /// Sets the pipeline name if source is MAX Graph.
  void setPipelineName(std::string name) { pipelineName = name; }

  /// Sets the path for the model's CAS config.  This is set from the public
  /// API: `M_setModelPath`.
  void setModelPath(std::filesystem::path path) {
    // Do not overwrite existing values.
    assert(std::holds_alternative<std::monostate>(modelSource));
    modelSource = path;
  }

  /// Adds a path specifying the location of additional kernel libraries that
  /// should be loaded
  void addKernelLibraryPath(std::filesystem::path path) {
    passthroughConfigs.emplace_back(
        std::make_unique<KernelLibraryDriverConfig>(std::move(path)));
  }

  /// Enables (or disables) compiling the graph under the tile-based programming
  /// model. Set from the public API via the `tile_based_fusion` compile option.
  void setTileBasedFusion(bool tileBasedFusion) {
    // Remove any existing tile-based-fusion config so the latest value wins.
    llvm::erase_if(passthroughConfigs,
                   [](const std::unique_ptr<DriverConfig> &ptr) {
                     return isa<TileBasedFusionDriverConfig>(ptr.get());
                   });
    passthroughConfigs.emplace_back(
        std::make_unique<TileBasedFusionDriverConfig>(tileBasedFusion));
  }

  void setDevices(ArrayRef<M::Engine::ConfiguredDevice> devices) {
    // Remove any existing device config.
    llvm::erase_if(passthroughConfigs,
                   [](const std::unique_ptr<DriverConfig> &ptr) {
                     return isa<DeviceConfig>(ptr.get());
                   });

    if (auto deviceConfig = selectDeviceConfig(devices))
      passthroughConfigs.push_back(std::move(deviceConfig));
  }

private:
  std::vector<std::unique_ptr<DriverConfig>> passthroughConfigs;
  std::variant<std::monostate, std::filesystem::path, ModelSource> modelSource;
  std::optional<std::string> pipelineName;
};

/// `UninitializedModel` encapsulates the collection of `M::Model` instances
/// that are loaded from the compiled MEF but not yet initialized.
struct UninitializedModel {
  UninitializedModel(M::BufferRef buffer, ModelList models,
                     RCRef<MEFFile> handle)
      : compiledModelBuffer(std::move(buffer)), models(std::move(models)),
        mefHandle(std::move(handle)) {}

  UninitializedModel(UninitializedModel &&other) = default;
  UninitializedModel &operator=(UninitializedModel &&other) = default;

  size_t getNumModels() const { return models.size(); }
  const M::Model *getFirstModel() const {
    return models.empty() ? nullptr : models.front().get();
  }
  const M::Model *getModel(size_t i) const {
    return i < models.size() ? models[i].get() : nullptr;
  }

  ArrayRef<std::unique_ptr<const Model>> getModels() const { return models; }

  M::BufferRef getBuffer() { return compiledModelBuffer; }

private:
  UninitializedModel(const UninitializedModel &) = delete;
  void operator=(const UninitializedModel &) = delete;

  M::BufferRef compiledModelBuffer;
  ModelList models;
  RCRef<MEFFile> mefHandle;
};

/// The internal type created when compiling one or more models from the public
/// API: `M_compileModel`. `CompiledModels` contains information such as the
/// source model and all compiled models loaded to memory.
struct CompiledModels {

  CompiledModels(SourceModel sourceModel,
                 std::unique_ptr<MOFContext> frameworkCtx,
                 UninitializedModel uninitializedModel)
      : sourceModel(std::move(sourceModel)),
        frameworkCtx(std::move(frameworkCtx)),
        uninitializedModel(std::move(uninitializedModel)) {}

  CompiledModels(CompiledModels &&model) = default;

  /// Compiles a model from an opaque source representation with a known format.
  static ErrorOr<CompiledModels> create(M::ContextRef context,
                                        const CompilationRequest &compCfg,
                                        const M::Engine::Config &rtCfg);

  /// Reads previously compiled models from an in-memory MEF artifact.
  /// The buffer must satisfy `maxAttributeAlignment()` (file mappings
  /// always do).
  static MODULAR_CXX_EXPORT ErrorOr<CompiledModels> read(BufferRef buffer);

  const SourceModel &getSourceModel() const { return sourceModel; }
  size_t getNumModels() const { return uninitializedModel.getNumModels(); }
  const M::Model *getFirstModel() const {
    return uninitializedModel.getFirstModel();
  }
  const M::Model *getModel(size_t i) const {
    return uninitializedModel.getModel(i);
  }

  using iterator = ArrayRef<std::unique_ptr<const Model>>::iterator;

  /// Return the list references to each model.
  ArrayRef<std::unique_ptr<const Model>> getModels() const {
    return uninitializedModel.getModels();
  }

  /// Iterate over the list of models.
  iterator begin() const { return uninitializedModel.getModels().begin(); }
  iterator end() const { return uninitializedModel.getModels().end(); }

  M::BufferRef getBuffer() { return uninitializedModel.getBuffer(); }

private:
  static ErrorOr<CompiledModels> read(BufferRef buffer,
                                      std::unique_ptr<MOFContext> frameworkCtx,
                                      SourceModel sourceModel);

  SourceModel sourceModel;
  std::unique_ptr<MOFContext> frameworkCtx;
  UninitializedModel uninitializedModel;

  CompiledModels(const CompiledModels &other) = delete;
  void operator=(const CompiledModels &other) = delete;
};

} // namespace M

#endif // GENERICML_API_COMPILEDMODELS_H
