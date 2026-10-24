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
// This file defines MEF-backed model and execution interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MEFMODEL_MEFMODEL_H
#define GENERICML_MEFMODEL_MEFMODEL_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Support/Diagnostic.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "GraphCompiler/Driver/Device/DeviceMap.h"
#include "GraphCompiler/Support/QuantSpec.h"
#include "GraphCompiler/Support/ResultInfo.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/DebugPrint.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace M {

class MEFInvokable;
class PrimitiveDescriptor;
class ModelSpec;

/// A device-resident input buffer and its tensor spec, as passed across the
/// capture/replay/execute boundary.
struct ModelInputBuffer {
  Driver::DeviceBufferRef buffer;
  TensorSpec spec;
};

//===----------------------------------------------------------------------===//
// ExecutableModel
//===----------------------------------------------------------------------===//

/// Created by Model::setup; executable many times without reinitializing.
class MODULAR_CXX_EXPORT ExecutableModel {
public:
  ExecutableModel(ExecutableModel &&) = default;
  ~ExecutableModel();

  const class Model *getModel() const { return model; }
  CPUDevice &getCPUDevice() const { return cpuDevice; }

  /// Wait for model setup to finish.
  ErrorOrSuccess waitForModelSetup();

  /// Runs the model on `args`, taking ownership. Returns the results, chain
  /// first, without awaiting them.
  SmallVector<AnyAsyncValueRef>
  execute(MutableArrayRef<AnyAsyncValueRef> args) const;

  ErrorOrSuccess printValue(raw_ostream &os, AnyAsyncValueRef const &value,
                            ResultOutputStyle style, unsigned precision) const;
  ErrorOrSuccess processResultForBinaryOutput(const AnyAsyncValueRef &result,
                                              const std::string &outputName,
                                              ProcessedResults &results) const;

  /// Capture a device graph collection. ``graphKeys`` holds one key
  /// (broadcast) or one per graph-capture device.
  ErrorOr<SmallVector<AnyAsyncValueRef>>
  capture(ArrayRef<uint64_t> graphKeys, ArrayRef<ModelInputBuffer> inputs);

  /// Replay a previously captured graph collection for per-device keys.
  ErrorOrSuccess replay(ArrayRef<uint64_t> graphKeys,
                        ArrayRef<ModelInputBuffer> inputs);

  /// Execute eagerly and compare the launch trace against the captured graphs.
  ErrorOrSuccess debugVerifyReplay(ArrayRef<uint64_t> graphKeys,
                                   ArrayRef<ModelInputBuffer> inputs);

  /// Await all pending device graph instantiations; call after a capture loop.
  ErrorOrSuccess awaitDeviceGraphs();

  /// Release captured device graphs for ``graphKeys``. Stream ordering frees
  /// them only after in-flight replays on the same stream. Uncaptured keys are
  /// no-ops; errors if the key count doesn't match the capture-capable
  /// streams, as ``capture``/``replay`` do.
  ErrorOrSuccess releaseCapturedGraphs(ArrayRef<uint64_t> graphKeys);

  ExecutableModel(const ExecutableModel &) = delete;
  ExecutableModel &operator=(const ExecutableModel &) = delete;

private:
  friend class Model; // Allow Model::setup() to construct

  ExecutableModel(AnyAsyncValueRef chain, AnyAsyncValueRef mgpContext,
                  AnyAsyncValueRef weightsRegistry,
                  SmallVector<AnyAsyncValueRef> &&extraArgs,
                  const class Model *model, CPUDevice &cpuDevice);

  struct CapturedPerDeviceGraphEntry;

  ErrorOr<SmallVector<AnyAsyncValueRef>>
  executeWithInputs(ArrayRef<ModelInputBuffer> inputs) const;

  ErrorOrSuccess
  validateCapturedPerDeviceKeys(ArrayRef<uint64_t> graphKeys) const;

  ErrorOrSuccess
  validateCapturedPerDeviceEntries(ArrayRef<uint64_t> graphKeys,
                                   ArrayRef<uint64_t> inputSignatures) const;

  ErrorOrSuccess launchDeviceGraphs(ArrayRef<uint64_t> normalizedKeys) const;

  const class Model *model;
  CPUDevice &cpuDevice;
  AnyAsyncValueRef chain;
  AnyAsyncValueRef mgpContext;
  AnyAsyncValueRef weightsRegistry;
  SmallVector<AnyAsyncValueRef> extraArgs;

  struct CapturedPerDeviceGraphEntry {
    Driver::AsyncDeviceGraphRef graph;
    // Device-local input fingerprint at capture time, distinct from graphKey,
    // used only to reject unsafe graphKey reuse.
    uint64_t inputSignature = 0;
  };

  /// Per-device captured graphs keyed by caller-provided scalar graph key.
  SmallVector<llvm::DenseMap<uint64_t, CapturedPerDeviceGraphEntry>>
      perDeviceGraphs;
};

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

/// An executable model's signature and metadata, decoded from MEF.
class Model {
public:
  static constexpr CompiledFrameworkLabel kFrameworkLabel{
      CompiledFrameworkLabel::kModularModel};

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&) = default;
  Model &operator=(Model &&) = default;
  ~Model() = default;

  StringRef getName() const { return name; }
  const MEFInvokable *getInitFn() const { return initFn; }
  const MEFInvokable *getExecuteFn() const { return executeFn; }
  const EncodedLocation &getLocation() const { return location; }

  /// Extract dSummary strings from all mgp.generic.execute primitives in the
  /// model's execute function.
  std::vector<std::string> getKernelSummaries() const;

  /// Return information about the input tensors.
  size_t getNumInputs() const { return inputSpecs.size(); }
  ArrayRef<QuantTensorSpec> getInputSpecs() const { return inputSpecs; }

  /// Gets the `QuantTensorSpec` for the input tensor named `inputTensorName`.
  ErrorOr<QuantTensorSpec>
  getInputSpecByName(llvm::StringRef inputTensorName) const {
    auto internalTensorIndexOr = getInputTensorIndexFromName(inputTensorName);
    if (internalTensorIndexOr.isError())
      return internalTensorIndexOr.takeError();
    return inputSpecs[*internalTensorIndexOr];
  }

  /// Gets the `QuantTensorSpec` for the output tensor named `outputTensorName`.
  ErrorOr<QuantTensorSpec>
  getOutputSpecByName(llvm::StringRef outputTensorName) const {
    auto internalTensorIndexOr = getOutputTensorIndexFromName(outputTensorName);
    if (internalTensorIndexOr.isError())
      return internalTensorIndexOr.takeError();
    return outputSpecs[*internalTensorIndexOr];
  }

  /// In MEF order.
  ArrayRef<std::string> getInputNames() const { return inputNames; }

  /// In MEF order.
  ArrayRef<std::string> getOutputNames() const { return outputNames; }

  /// Return information about the output tensors.
  size_t getNumOutputs() const { return outputSpecs.size(); }
  ArrayRef<QuantTensorSpec> getOutputSpecs() const { return outputSpecs; }

  /// Initialize the model without blocking. `deviceMap` maps the
  /// compile-time `DeviceRef`s to device handles.
  AsyncRT::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
  setup(CPUDevice &cpuDevice, const WeightsRegistry *weightsRegistry = {},
        const M::GraphCompiler::Driver::DeviceMap &deviceMap =
            M::GraphCompiler::Driver::DeviceMap()) const;

  /// Return the devices used by the model.
  ArrayRef<DeviceRef> getDevices() const { return devices; }
  /// Return an array of devices that map 1:1 to the devices used by each input.
  ArrayRef<DeviceRef> getInputDevices() const { return inputDevices; }
  /// Return an array of devices that map 1:1 to the devices used by each
  /// output.
  ArrayRef<DeviceRef> getOutputDevices() const { return outputDevices; }

  static StringLiteral getEntityName() { return "mef.model"; }
  static ErrorOr<std::unique_ptr<Model>>
  getFromDescriptor(const PrimitiveDescriptor &descriptor);

  Model(std::string name, SmallVector<DeviceRef> devices,
        CompiledFrameworkLabel frameworkLabel, const MEFInvokable *initFn,
        const MEFInvokable *executeFn, ArrayRef<QuantTensorSpec> inputSpecs,
        ArrayRef<QuantTensorSpec> outputSpecs,
        SmallVector<DeviceRef> inputDevices,
        SmallVector<DeviceRef> outputDevices, EncodedLocation location);

  Model(std::string name, SmallVector<DeviceRef> devices,
        CompiledFrameworkLabel frameworkLabel, const MEFInvokable *initFn,
        const MEFInvokable *executeFn, ArrayRef<QuantTensorSpec> inputSpecs,
        ArrayRef<QuantTensorSpec> outputSpecs, ArrayRef<StringRef> inputNames,
        ArrayRef<StringRef> outputNames, SmallVector<DeviceRef> inputDevices,
        SmallVector<DeviceRef> outputDevices, EncodedLocation location);

  Model(const ModelSpec &spec);

  void initializeSignature(ArrayRef<QuantTensorSpec> inputSpecs,
                           ArrayRef<QuantTensorSpec> outputSpecs,
                           ArrayRef<StringRef> inputNames,
                           ArrayRef<StringRef> outputNames);

  /// Names of input and output tensors.
  SmallVector<std::string> inputNames;
  SmallVector<std::string> outputNames;

private:
  /// Internal index (not source order) of the named input.
  ErrorOr<size_t>
  getInputTensorIndexFromName(llvm::StringRef inputTensorName) const {
    if (auto it = inputTensorNameToIndex.find(inputTensorName);
        it != inputTensorNameToIndex.end())
      return it->second;

    return Error("Did not find input tensor by name " + inputTensorName);
  }

  /// Internal index (not source order) of the named output.
  ErrorOr<size_t>
  getOutputTensorIndexFromName(llvm::StringRef outputTensorName) const {
    if (auto it = outputTensorNameToIndex.find(outputTensorName);
        it != outputTensorNameToIndex.end())
      return it->second;

    return Error("Did not find output tensor by name " + outputTensorName);
  }

  std::string name;
  const MEFInvokable *initFn = nullptr;
  const MEFInvokable *executeFn = nullptr;
  SmallVector<DeviceRef> devices;
  SmallVector<DeviceRef> inputDevices;
  SmallVector<DeviceRef> outputDevices;
  EncodedLocation location =
      AsyncRT::UnknownLocationDecoder::getEncodedLocation();

  /// The model's signature.
  SmallVector<QuantTensorSpec> inputSpecs;
  SmallVector<QuantTensorSpec> outputSpecs;

  /// The map is used for locating the input/output spec by the name.
  llvm::StringMap<size_t> inputTensorNameToIndex;
  llvm::StringMap<size_t> outputTensorNameToIndex;
};

/// Container for MEF-decoded models.
using ModelList = SmallVector<std::unique_ptr<const Model>>;

/// Load all models from an MEF file, which must outlive them.
DRIVER_VISIBILITY_EXPORT // for MT
    ErrorOr<ModelList> loadMGPMEFFile(RCRef<MEFFile> mefFile);

} // namespace M

#endif // GENERICML_MEFMODEL_MEFMODEL_H
