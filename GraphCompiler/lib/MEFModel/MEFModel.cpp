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

#include "GraphCompiler/MEFModel/MEFModel.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "GraphCompiler/MGPDialect/Primitives/MGPPrimitives.h"
#include "GraphCompiler/Support/FormatYAML.h"
#include "GraphCompiler/Support/MEFTypeUtils.h"
#include "GraphCompiler/Support/ModelSpec.h"
#include "GraphCompiler/Support/QuantSpec.h"
#include "GraphCompiler/Support/ResultInfo.h"
#include "GraphCompiler/Support/Value.h"
#include "MLRT/Core/MojoValue.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "MLRT/Driver/DeviceContext/KernelLoadGate.h"
#include "MLRT/GraphRT/MEF/MEFAttr.h"
#include "MLRT/GraphRT/MEF/MEFTypes.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveDescriptor.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "MLRT/GraphRT/Primitives.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/HashUtils.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <vector>

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;

static SmallVector<QuantTensorSpec>
getSpecsAsVector(ArrayRef<ModelSpec::ModelIOSpec> modelIOSpec) {
  SmallVector<QuantTensorSpec> result(modelIOSpec.size());
  llvm::transform(
      modelIOSpec, result.begin(),
      [](const ModelSpec::ModelIOSpec &ioSpec) { return ioSpec.spec; });
  return result;
}

static SmallVector<StringRef>
getNamesAsVector(ArrayRef<ModelSpec::ModelIOSpec> modelIOSpec) {
  SmallVector<StringRef> result(modelIOSpec.size());
  llvm::transform(modelIOSpec, result.begin(),
                  [](const ModelSpec::ModelIOSpec &ioSpec) {
                    return StringRef(ioSpec.name);
                  });
  return result;
}

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

Model::Model(const ModelSpec &spec)
    : name(spec.name),
      location(AsyncRT::UnknownLocationDecoder::getEncodedLocation()) {
  auto inputSpecs = getSpecsAsVector(spec.inputs);
  auto outputSpecs = getSpecsAsVector(spec.outputs);
  auto inputNames = getNamesAsVector(spec.inputs);
  auto outputNames = getNamesAsVector(spec.outputs);
  initializeSignature(inputSpecs, outputSpecs, inputNames, outputNames);
}

void Model::initializeSignature(ArrayRef<QuantTensorSpec> inputSpecs,
                                ArrayRef<QuantTensorSpec> outputSpecs,
                                ArrayRef<StringRef> inputNamesRefs,
                                ArrayRef<StringRef> outputNamesRefs) {
  inputNames.clear();
  outputNames.clear();
  inputTensorNameToIndex.clear();
  outputTensorNameToIndex.clear();

  if (inputNamesRefs.empty()) {
    for (size_t i = 0; i < inputSpecs.size(); ++i)
      inputNames.push_back("input" + std::to_string(i));
  } else {
    inputNames.reserve(inputNamesRefs.size());
    for (StringRef nameRef : inputNamesRefs)
      inputNames.emplace_back(nameRef.str());
  }

  if (outputNamesRefs.empty()) {
    for (size_t i = 0; i < outputSpecs.size(); ++i)
      outputNames.push_back("output" + std::to_string(i));
  } else {
    outputNames.reserve(outputNamesRefs.size());
    for (StringRef nameRef : outputNamesRefs)
      outputNames.emplace_back(nameRef.str());
  }

  this->inputSpecs.assign(inputSpecs.begin(), inputSpecs.end());
  this->outputSpecs.assign(outputSpecs.begin(), outputSpecs.end());

  for (size_t i = 0, e = getNumInputs(); i < e; ++i)
    inputTensorNameToIndex.insert({inputNames[i], i});

  for (size_t i = 0, e = getNumOutputs(); i < e; ++i)
    outputTensorNameToIndex.insert({outputNames[i], i});
}

Model::Model(std::string name, SmallVector<DeviceRef> devices,
             CompiledFrameworkLabel frameworkLabel, const MEFInvokable *initFn,
             const MEFInvokable *executeFn,
             ArrayRef<QuantTensorSpec> inputSpecs,
             ArrayRef<QuantTensorSpec> outputSpecs,
             SmallVector<DeviceRef> inputDevices,
             SmallVector<DeviceRef> outputDevices, EncodedLocation location)
    : name(std::move(name)), initFn(initFn), executeFn(executeFn),
      devices(std::move(devices)), inputDevices(std::move(inputDevices)),
      outputDevices(std::move(outputDevices)), location(std::move(location)) {
  ASSERT_STREAM(frameworkLabel == kFrameworkLabel,
                << "Expected '" << kFrameworkLabel.getAsString()
                << "' label for model, but got '"
                << frameworkLabel.getAsString() << "'");
  initializeSignature(inputSpecs, outputSpecs, {}, {});
}

Model::Model(std::string name, SmallVector<DeviceRef> devices,
             CompiledFrameworkLabel frameworkLabel, const MEFInvokable *initFn,
             const MEFInvokable *executeFn,
             ArrayRef<QuantTensorSpec> inputSpecs,
             ArrayRef<QuantTensorSpec> outputSpecs,
             ArrayRef<StringRef> inputNames, ArrayRef<StringRef> outputNames,
             SmallVector<DeviceRef> inputDevices,
             SmallVector<DeviceRef> outputDevices, EncodedLocation location)
    : name(std::move(name)), initFn(initFn), executeFn(executeFn),
      devices(std::move(devices)), inputDevices(std::move(inputDevices)),
      outputDevices(std::move(outputDevices)), location(std::move(location)) {
  ASSERT_STREAM(frameworkLabel == kFrameworkLabel,
                << "Expected " << kFrameworkLabel.getAsString()
                << " label for model, but got "
                << frameworkLabel.getAsString());
  initializeSignature(inputSpecs, outputSpecs, inputNames, outputNames);
}

static ErrorOrSuccess
ensureStreamCountForDevices(ArrayRef<AnyAsyncValueRef> deviceContextArgs,
                            int64_t streamCount) {
  if (streamCount <= 1)
    return success();

  for (const AnyAsyncValueRef &deviceContextArg : deviceContextArgs) {
    auto *ctxRefPtr = dyn_cast<Driver::DeviceContextRef>(deviceContextArg);
    if (!ctxRefPtr || !ctxRefPtr->getPointer())
      return Error("DeviceContextRef is null");
    Driver::DeviceContextRef ctxRef = *ctxRefPtr;
    // Only CUDA/HIP device contexts support multi-stream.
    Driver::Device::API api = ctxRef->device().api();
    if (api != Driver::Device::cudaAPI && api != Driver::Device::hipAPI)
      continue;
    // Grow the context's stream pool until it holds `streamCount` streams.
    for (unsigned i = ctxRef->numStreams(); i < streamCount; ++i) {
      if (auto errorOrStream = ctxRef->createStream())
        return errorOrStream.takeError();
    }
  }

  return success();
}

AsyncRT::ErrorDiagnosticOr<std::unique_ptr<ExecutableModel>>
Model::setup(CPUDevice &cpuDevice, const WeightsRegistry *weightsRegistry,
             const M::GraphCompiler::Driver::DeviceMap &deviceMap) const {
  checkUniqueCPUDevice(cpuDevice);

  SmallVector<AnyAsyncValueRef> initArgs;
  initArgs.reserve(kMgpModelNumFixedImplicitInitArgs + devices.size());

  // Borrowed.
  initArgs.push_back(AnyAsyncValueRef::createReady<MojoValue>(
      cpuDevice, MojoValue::createRef(
                     weightsRegistry ? weightsRegistry->opaque() : nullptr)));

  auto deviceContextsOr = GraphCompiler::Driver::mapDeviceRefsToDevices(
      getDevices(), deviceMap, cpuDevice);
  if (deviceContextsOr.isError())
    return AsyncRT::EncodedDiagnostic(
        Twine("setup for model ") + getName() +
            " failed to map DeviceRefs to Devices: " +
            Twine(deviceContextsOr.getError()),
        location.copy());
  initArgs.append(std::make_move_iterator(deviceContextsOr->begin()),
                  std::make_move_iterator(deviceContextsOr->end()));

  // Pre-create side streams before any capture begins. Defaults to 2 (base +
  // one side); see the device-context.stream-count config key.
  const int64_t streamCount =
      Driver::DeviceContext::configAsInt("stream-count", 2);
  ErrorOrSuccess streamError = ensureStreamCountForDevices(
      llvm::ArrayRef(initArgs).take_back(getDevices().size()), streamCount);
  if (streamError.isError())
    return AsyncRT::EncodedDiagnostic(
        Twine("setup for model ") + getName() +
            " failed to create device streams: " + streamError.getError(),
        location.copy());

  auto results = initFn->invoke(initArgs, cpuDevice);

  for (AnyAsyncValueRef &result : results) {
    if (result.isError()) {
      auto diagnostic = result.takeDiagnostic();
      return AsyncRT::EncodedDiagnostic(
          Twine("setup for model ") + getName() +
              " returned error: " + std::string(diagnostic.getMessage().get()),
          diagnostic.getLocation().copy());
    }
  }

  SmallVector<AnyAsyncValueRef> extraArgs;
  extraArgs.reserve(results.size() - kMgpModelNumFixedImplicitExecArgs);
  for (size_t i = kMgpModelNumFixedImplicitExecArgs; i < results.size(); ++i)
    extraArgs.push_back(std::move(results[i]));

  return {std::unique_ptr<ExecutableModel>(new ExecutableModel(
      /*chain=*/std::move(results[0]),
      /*mgpContext=*/std::move(results[1]),
      /*weightsRegistry=*/std::move(results[2]),
      /*extraArgs*/ std::move(extraArgs), this, cpuDevice))};
}

ErrorOr<std::unique_ptr<Model>>
Model::getFromDescriptor(const PrimitiveDescriptor &desc) {
  constexpr size_t devicesAttrIdx = 0;
  constexpr size_t frameworkNameAttrIdx = 1;
  constexpr size_t modelFnTypeAttrIdx = 4;
  constexpr size_t inputDeviceIndicesIdx = 5;
  constexpr size_t resultDeviceIndicesIdx = 6;
  constexpr size_t argumentNamesIdx = 7;
  constexpr size_t resultNamesIdx = 8;
  constexpr size_t modelNameAttrIdx = 9;

  constexpr size_t initFnIdx = 0;
  constexpr size_t executeFnIdx = 1;

  StringRef modelName = *desc.getAttr<StringRef>(modelNameAttrIdx);
  StringRef frameworkName = *desc.getAttr<StringRef>(frameworkNameAttrIdx);

  CompiledFrameworkLabel frameworkLabel =
      CompiledFrameworkLabel::getLabelForOpName(getEntityName(), frameworkName);

  MEFFunctionType signature = desc.getAttr<MEFFunctionType>(modelFnTypeAttrIdx);

  auto devicesJson = *desc.getAttr<StringRef>(devicesAttrIdx);
  auto valueOr = llvm::json::parse(devicesJson);
  if (!valueOr)
    return toModularError(valueOr.takeError());

  llvm::SmallVector<DeviceRef> devices(valueOr->getAsArray()->size());
  for (auto [v, dev] : llvm::zip(*valueOr->getAsArray(), devices)) {
    auto deviceOr = DeviceRef::deserializeFromJSON(&v);
    if (deviceOr.isError())
      return deviceOr.takeError();
    dev = deviceOr.takeValue();
  }

  const MEFInvokable *init = desc.getInvokable(initFnIdx);
  const MEFInvokable *execute = desc.getInvokable(executeFnIdx);

  SmallVector<QuantTensorSpec> inputSpecs, outputSpecs;
  if (auto err = decodeQuantizedSignature(signature, inputSpecs, outputSpecs))
    return err.takeError();

  auto getIODevices = [&](const PrimitiveDescriptor &desc,
                          size_t attrIdx) -> llvm::SmallVector<DeviceRef> {
    auto deviceList = desc.getAttr<MEFListAttr>(attrIdx);
    SmallVector<DeviceRef> ioDevices;
    for (MEFDynamicAttr attr : deviceList)
      ioDevices.push_back(devices[attr.cast<MEFAttr<uint32_t>>().get()]);

    return ioDevices;
  };

  auto getNames = [&](const PrimitiveDescriptor &desc,
                      size_t attrIdx) -> llvm::SmallVector<StringRef> {
    auto namesAttr = desc.getAttr<MEFListAttr>(attrIdx);
    SmallVector<StringRef> namesList;
    for (MEFDynamicAttr attr : namesAttr) {
      namesList.push_back(attr.cast<MEFAttr<StringRef>>().get());
    }
    return namesList;
  };

  SmallVector<DeviceRef> inputDevices =
      getIODevices(desc, inputDeviceIndicesIdx);
  SmallVector<DeviceRef> outputDevices =
      getIODevices(desc, resultDeviceIndicesIdx);

  SmallVector<StringRef> inputNames = getNames(desc, argumentNamesIdx);
  SmallVector<StringRef> outputNames = getNames(desc, resultNamesIdx);

  return std::make_unique<Model>(modelName.str(), devices, frameworkLabel, init,
                                 execute, inputSpecs, outputSpecs, inputNames,
                                 outputNames, inputDevices, outputDevices,
                                 desc.copyLocation());
}

std::vector<std::string> Model::getKernelSummaries() const {
  assert(executeFn && "getKernelSummaries requires a compiled model with an "
                      "execute function");

  constexpr size_t kSummaryAttrIndex = 3;

  // TODO(GEX-3543): This only traverses the execute region. Kernels in
  // subgraph regions and the init region are not included.
  std::vector<std::string> summaries;
  auto result = executeFn->recursePrimitives(
      MEFInvokable::kPreOrder,
      [&](StringRef primitiveName,
          const PrimitiveDescriptor &desc) -> ErrorOrSuccess {
        if (primitiveName == "mgp.generic.execute" &&
            desc.getNumAttrs() > kSummaryAttrIndex) {
          const uint8_t *raw = desc.getRawAttr(kSummaryAttrIndex);
          size_t len = decodeMEFArrayLength(raw);
          summaries.emplace_back(reinterpret_cast<const char *>(raw), len);
        }
        return success();
      });
  (void)result; // Visitor always succeeds.

  return summaries;
}

//===----------------------------------------------------------------------===//
// ExecutableModel
//===----------------------------------------------------------------------===//

ExecutableModel::ExecutableModel(AnyAsyncValueRef chain,
                                 AnyAsyncValueRef mgpContext,
                                 AnyAsyncValueRef weightsRegistry,
                                 SmallVector<AnyAsyncValueRef> &&extraArgs,
                                 const Model *model, CPUDevice &cpuDevice)
    : model(model), cpuDevice(cpuDevice), chain(std::move(chain)),
      mgpContext(std::move(mgpContext)),
      weightsRegistry(std::move(weightsRegistry)),
      extraArgs(std::move(extraArgs)) {}

ExecutableModel::~ExecutableModel() = default;

SmallVector<AnyAsyncValueRef>
ExecutableModel::execute(MutableArrayRef<AnyAsyncValueRef> args) const {
  TRACY_ZONE_SCOPED_NC("ExecutableModel::execute", TRACY_COLOR_TEAL);

  checkUniqueCPUDevice(getCPUDevice());

  // Fixed args (chain, context, weights registry), devices, init->exec context
  // bundles, then inputs. See ModelOp in MGPOps.td.
  SmallVector<AnyAsyncValueRef> passedArgs;
  passedArgs.reserve(kMgpModelNumFixedImplicitExecArgs + extraArgs.size() +
                     args.size());
  passedArgs.emplace_back(chain.copy());
  passedArgs.emplace_back(mgpContext.copy());
  passedArgs.emplace_back(weightsRegistry.copy());
  for (const AnyAsyncValueRef &dev : extraArgs)
    passedArgs.emplace_back(dev.copy());
  for (AnyAsyncValueRef &arg : args)
    passedArgs.emplace_back(std::move(arg));

  // Some kernels read the thread-local cpuDevice.
  AsyncRT::CompactCPUDevicePtr::setCurrentCPUDevice(getCPUDevice());

  return getModel()->getExecuteFn()->invoke(passedArgs, getCPUDevice());
}

ErrorOrSuccess ExecutableModel::waitForModelSetup() {
  AsyncRT::await({chain.copy(), mgpContext.copy(), weightsRegistry.copy()});
  AsyncRT::await(extraArgs);
  if (chain.isError())
    return Error(chain.takeDiagnostic().getMessage().get());
  if (mgpContext.isError())
    return Error(mgpContext.takeDiagnostic().getMessage().get());
  if (weightsRegistry.isError())
    return Error(weightsRegistry.takeDiagnostic().getMessage().get());
  for (auto &arg : extraArgs) {
    if (arg.isError())
      return Error(arg.takeDiagnostic().getMessage().get());
  }
  return success();
}

// Pinned host memory or a CPU context. Mirrors GenericML DeviceMemory::isHost.
static bool isHostResident(const Driver::DeviceBufferRef &buffer) {
  return buffer->isHost() ||
         buffer->context().device().api() == Driver::Device::cpuAPI;
}

// Stable hash of (index, address, shape, dtype) over the non-host inputs on
// this device only, so replay can mix device keys from different captures.
static uint64_t computeInputKeyForDevice(ArrayRef<ModelInputBuffer> inputs,
                                         const Driver::Device &device) {
  raw_xxhash64_stream hasher;

  uint64_t deviceId = static_cast<uint64_t>(device.id());
  hasher.write(reinterpret_cast<const char *>(&deviceId), sizeof(deviceId));

  for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
    const ModelInputBuffer &input = inputs[inputIndex];
    if (isHostResident(input.buffer))
      continue;

    const Driver::Device &inputDevice = input.buffer->context().device();
    if (inputDevice.id() != device.id() || inputDevice.api() != device.api())
      continue;

    uint64_t stableInputIndex = static_cast<uint64_t>(inputIndex);
    hasher.write(reinterpret_cast<const char *>(&stableInputIndex),
                 sizeof(stableInputIndex));

    void *bufferAddress = input.buffer->ptr();
    hasher.write(reinterpret_cast<const char *>(&bufferAddress),
                 sizeof(bufferAddress));

    const TensorSpec &spec = input.spec;
    uint64_t rank = spec.getRank();
    hasher.write(reinterpret_cast<const char *>(&rank), sizeof(rank));
    for (int64_t dim : spec)
      hasher.write(reinterpret_cast<const char *>(&dim), sizeof(dim));

    uint64_t dtype = static_cast<uint64_t>(spec.getEltType().getValue());
    hasher.write(reinterpret_cast<const char *>(&dtype), sizeof(dtype));
  }
  return hasher.hash();
}

static SmallVector<uint64_t>
computePerDeviceInputSignatures(ArrayRef<ModelInputBuffer> inputs,
                                ArrayRef<Driver::DeviceStreamRef> streams) {
  SmallVector<uint64_t> signatures;
  signatures.reserve(streams.size());
  for (const Driver::DeviceStreamRef &stream : streams)
    signatures.push_back(computeInputKeyForDevice(inputs, stream->device()));
  return signatures;
}

static std::string graphKeysToString(ArrayRef<uint64_t> graphKeys);

static ErrorOr<SmallVector<uint64_t>>
normalizeGraphKeysForDevices(ArrayRef<uint64_t> graphKeys, size_t deviceCount,
                             StringRef operationName) {
  if (deviceCount == 0)
    return Error("No graph-capture devices available for this model");

  if (graphKeys.empty())
    return Error(Twine("graph keys must not be empty for ") + operationName);

  if (graphKeys.size() == 1 && deviceCount > 1)
    return SmallVector<uint64_t>(deviceCount, graphKeys.front());

  if (graphKeys.size() != deviceCount) {
    std::string keysString = graphKeysToString(graphKeys);
    return Error(Twine("per-device graph key count mismatch for ") +
                 operationName + ": got " + Twine(graphKeys.size()) + " keys " +
                 keysString + ", expected " + Twine(deviceCount));
  }

  return SmallVector<uint64_t>(graphKeys.begin(), graphKeys.end());
}

static std::string graphKeysToString(ArrayRef<uint64_t> graphKeys) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "[";
  llvm::interleaveComma(graphKeys, os);
  os << "]";
  return os.str();
}

// Graph-capture-capable streams of the model's device contexts, so
// capture/replay use the same devices as execution.
static ErrorOr<SmallVector<Driver::DeviceStreamRef>>
captureStreams(MutableArrayRef<AnyAsyncValueRef> extraArgs) {
  if (auto err = AsyncRT::awaitOrError(extraArgs))
    return err.takeError();

  SmallVector<Driver::DeviceStreamRef> results;
  for (const AnyAsyncValueRef &arg : extraArgs) {
    M::RCRef<M::Driver::DeviceContext> *context =
        dyn_cast<Driver::DeviceContextRef>(arg);
    if (!context)
      continue;

    assert(context->getPointer() != nullptr &&
           "captureStreams: DeviceContextRef holds a null pointer — "
           "was mapDeviceRefsToDevices called correctly?");
    Driver::Device::API api = context->getPointer()->device().api();
    if (api != Driver::Device::cudaAPI && api != Driver::Device::hipAPI)
      continue;

    ErrorOr<Driver::DeviceStreamRef> streamOr = context->getPointer()->stream();
    if (streamOr.isError())
      return streamOr.takeError();

    results.push_back(streamOr.takeValue());
  }

  return results;
}

ErrorOr<SmallVector<AnyAsyncValueRef>>
ExecutableModel::executeWithInputs(ArrayRef<ModelInputBuffer> inputs) const {
  SmallVector<AnyAsyncValueRef> inputValues;
  inputValues.reserve(inputs.size());
  for (const ModelInputBuffer &input : inputs) {
    inputValues.push_back(GraphCompiler::Driver::prepareInput(
        getCPUDevice(), input.buffer, input.spec));
  }

  SmallVector<AnyAsyncValueRef> results = execute(inputValues);
  if (auto err = AsyncRT::awaitOrError(results))
    return err.takeError();

  return results;
}

ErrorOr<SmallVector<AnyAsyncValueRef>>
ExecutableModel::capture(ArrayRef<uint64_t> graphKeys,
                         ArrayRef<ModelInputBuffer> inputs) {
  ErrorOr<SmallVector<Driver::DeviceStreamRef>> streamsOr =
      captureStreams(extraArgs);

  if (streamsOr.isError())
    return streamsOr.takeError();

  const size_t deviceCount = streamsOr->size();

  ErrorOr<SmallVector<uint64_t>> normalizedKeysOr =
      normalizeGraphKeysForDevices(graphKeys, deviceCount, "capture");
  if (normalizedKeysOr.isError())
    return normalizedKeysOr.takeError();
  ArrayRef<uint64_t> normalizedKeys = *normalizedKeysOr;

  if (perDeviceGraphs.empty())
    perDeviceGraphs.resize(deviceCount);
  else if (perDeviceGraphs.size() != deviceCount)
    return Error(Twine("capture device count changed from ") +
                 Twine(perDeviceGraphs.size()) + " to " + Twine(deviceCount));

  bool anyDeviceHit = false;
  for (size_t deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
    if (perDeviceGraphs[deviceIdx].contains(normalizedKeys[deviceIdx])) {
      anyDeviceHit = true;
      break;
    }
  }

  if (anyDeviceHit) {
    std::string keysString = graphKeysToString(normalizedKeys);
    return Error(Twine("Graph keys already captured for capture: ") +
                 keysString + ". Use replay() for existing keys.");
  }

  // Only for per-device graph-key safety validation.
  SmallVector<uint64_t> inputSignatures =
      computePerDeviceInputSignatures(inputs, *streamsOr);

  SmallVector<AnyAsyncValueRef> graphResults;

  ErrorOr<Driver::DeviceGraphCollection> graphOrErr =
      Driver::DeviceGraph::capture(
          cpuDevice, *streamsOr, [&]() -> ErrorOrSuccess {
            auto maybeResult = executeWithInputs(inputs);
            if (maybeResult.isError())
              return maybeResult.takeError();

            graphResults = maybeResult.takeValue();
            return success();
          });

  if (graphOrErr.isError())
    return Error(Twine("Failed to capture graph: ") + graphOrErr.getError());

  // Don't propagate yet: materialize capture results before extracting the
  // graphs, even on error.
  ErrorOrSuccess err = AsyncRT::awaitOrError(graphResults);

  if (err.isError())
    return err.takeError();

  Driver::DeviceGraphCollection graphs = graphOrErr.takeValue();

  ArrayRef<Driver::AsyncDeviceGraphRef> capturedGraphs = graphs.getGraphs();
  if (capturedGraphs.size() != deviceCount) {
    return Error(Twine("captured graph count mismatch: expected ") +
                 Twine(deviceCount) + ", got " + Twine(capturedGraphs.size()));
  }

  for (size_t deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
    CapturedPerDeviceGraphEntry entry;
    entry.graph = capturedGraphs[deviceIdx].copy();
    entry.inputSignature = inputSignatures[deviceIdx];
    perDeviceGraphs[deviceIdx].try_emplace(normalizedKeys[deviceIdx],
                                           std::move(entry));
  }

  return graphResults;
}

ErrorOrSuccess ExecutableModel::validateCapturedPerDeviceKeys(
    ArrayRef<uint64_t> graphKeys) const {
  if (perDeviceGraphs.size() != graphKeys.size()) {
    return Error(Twine("capture device count changed from ") +
                 Twine(perDeviceGraphs.size()) + " to " +
                 Twine(graphKeys.size()));
  }

  for (size_t deviceIdx = 0; deviceIdx < graphKeys.size(); ++deviceIdx) {
    const auto &deviceMap = perDeviceGraphs[deviceIdx];
    auto it = deviceMap.find(graphKeys[deviceIdx]);
    if (it == deviceMap.end()) {
      return Error(Twine("No captured graph for device ") + Twine(deviceIdx) +
                   " key " + Twine(graphKeys[deviceIdx]));
    }
  }

  return success();
}

ErrorOrSuccess ExecutableModel::validateCapturedPerDeviceEntries(
    ArrayRef<uint64_t> graphKeys, ArrayRef<uint64_t> inputSignatures) const {
  if (ErrorOrSuccess err = validateCapturedPerDeviceKeys(graphKeys))
    return err.takeError();

  if (inputSignatures.size() != graphKeys.size()) {
    return Error(Twine("input signature count mismatch: expected ") +
                 Twine(graphKeys.size()) + ", got " +
                 Twine(inputSignatures.size()));
  }

  for (size_t deviceIdx = 0; deviceIdx < graphKeys.size(); ++deviceIdx) {
    const auto &deviceMap = perDeviceGraphs[deviceIdx];
    auto it = deviceMap.find(graphKeys[deviceIdx]);
    ASSERT_STREAM(it != deviceMap.end(), "expected captured graph entry");

    if (it->second.inputSignature != inputSignatures[deviceIdx]) {
      return Error(Twine("Unsafe graph key reuse on device ") +
                   Twine(deviceIdx) + " for key " +
                   Twine(graphKeys[deviceIdx]) + ": expected input signature " +
                   Twine(it->second.inputSignature) + ", got " +
                   Twine(inputSignatures[deviceIdx]) +
                   ". Use distinct caller-provided keys for distinct input "
                   "buffer/signature configurations.");
    }
  }

  return success();
}

// The gate for the backend this model runs on, or null if it has none. A host
// context has no gate, so take the first context that does; the accelerators
// share one driver.
static Driver::KernelLoadGate *
kernelLoadGateFor(ArrayRef<AsyncRT::AnyAsyncValueRef> extraArgs) {
  for (const AsyncRT::AnyAsyncValueRef &arg : extraArgs) {
    auto *ctxRef = dyn_cast<Driver::DeviceContextRef>(arg);
    if (ctxRef == nullptr || ctxRef->getPointer() == nullptr)
      continue;
    if (Driver::KernelLoadGate *gate = (*ctxRef)->kernelLoadGate())
      return gate;
  }
  return nullptr;
}

// Like MLRT::launchDeviceCollective, but dispatches only once the collective
// owns `gate`, and releases it when the last element has run. The caller holds
// nothing while it awaits the result, since a task the wait runs may need the
// gate.
template <typename ElementFn>
static AsyncRT::AsyncValueRef<AsyncRT::Chain>
launchGatedDeviceCollective(AsyncRT::CPUDevice &cpuDevice,
                            Driver::KernelLoadGate *gate, size_t count,
                            ElementFn elementFn) {
  auto dispatch = [&cpuDevice, gate, count,
                   elementFn = std::move(elementFn)]() mutable {
    AsyncRT::AsyncValueRef<AsyncRT::Chain> fanout =
        MLRT::launchDeviceCollective(cpuDevice, count, std::move(elementFn));
    if (gate != nullptr)
      fanout.andThenSync([gate] { gate->releaseCollective(); });
    return fanout;
  };

  AsyncRT::AsyncValueRef<AsyncRT::Chain> grant;
  if (gate == nullptr || gate->tryAcquireCollective(cpuDevice, grant))
    return dispatch();

  auto done = AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(cpuDevice);
  grant.andThenAsync(
      [dispatch = std::move(dispatch), done = done.copy()]() mutable {
        AsyncRT::AsyncValueRef<AsyncRT::Chain> fanout = dispatch();
        fanout.andThenSync(
            [done = std::move(done)]() mutable { std::move(done).emplace(); });
      });
  return done;
}

ErrorOrSuccess
ExecutableModel::launchDeviceGraphs(ArrayRef<uint64_t> normalizedKeys) const {
  if (normalizedKeys.empty())
    return success();

  SmallVector<ErrorOrSuccess, 8> errors(normalizedKeys.size());

  SmallVector<const CapturedPerDeviceGraphEntry *, 8> entries(
      normalizedKeys.size());
  for (size_t deviceIdx = 0; deviceIdx < normalizedKeys.size(); ++deviceIdx) {
    const auto &deviceMap = perDeviceGraphs[deviceIdx];
    auto it = deviceMap.find(normalizedKeys[deviceIdx]);
    ASSERT_STREAM(it != deviceMap.end(), "expected captured graph entry");
    entries[deviceIdx] = &it->second;
  }

  // Await every graph before any of them replays, so the replays themselves
  // never wait: a graph that never materializes would otherwise leave the
  // peers that did replay spinning at a barrier it will never reach.
  for (size_t deviceIdx = 0; deviceIdx < entries.size(); ++deviceIdx) {
    if (ErrorOrSuccess err = AsyncRT::awaitOrError(entries[deviceIdx]->graph))
      return err.takeError();
  }

  auto launchGraph = [&](size_t deviceIdx) {
    if (ErrorOrSuccess err = (*entries[deviceIdx]->graph)->replay())
      errors[deviceIdx] = err.takeError();
  };

  // A captured graph can contain collective kernels, so the gate keeps kernel
  // loads out until every replay has launched; see Driver::KernelLoadGate. A
  // single graph has no peers to strand, so it skips the gate.
  if (normalizedKeys.size() == 1) {
    launchGraph(0);
  } else {
    // launchGraph captures this frame by reference, which is sound because
    // this frame outlives the await even when a task the await runs buries it.
    AsyncRT::await(launchGatedDeviceCollective(
        cpuDevice, kernelLoadGateFor(extraArgs), normalizedKeys.size(),
        std::move(launchGraph)));
  }

  for (ErrorOrSuccess &err : errors) {
    if (err)
      return err.takeError();
  }

  return success();
}

ErrorOrSuccess ExecutableModel::replay(ArrayRef<uint64_t> graphKeys,
                                       ArrayRef<ModelInputBuffer> inputs) {
  ErrorOr<SmallVector<Driver::DeviceStreamRef>> streamsOr =
      captureStreams(extraArgs);

  if (streamsOr.isError())
    return streamsOr.takeError();

  ErrorOr<SmallVector<uint64_t>> normalizedKeysOr =
      normalizeGraphKeysForDevices(graphKeys, streamsOr->size(), "replay");
  if (normalizedKeysOr.isError())
    return normalizedKeysOr.takeError();
  ArrayRef<uint64_t> normalizedKeys = *normalizedKeysOr;

  SmallVector<uint64_t> inputSignatures =
      computePerDeviceInputSignatures(inputs, *streamsOr);

  if (ErrorOrSuccess err =
          validateCapturedPerDeviceEntries(normalizedKeys, inputSignatures))
    return err.takeError();

  return launchDeviceGraphs(normalizedKeys);
}

ErrorOrSuccess
ExecutableModel::debugVerifyReplay(ArrayRef<uint64_t> graphKeys,
                                   ArrayRef<ModelInputBuffer> inputs) {
  ErrorOr<SmallVector<Driver::DeviceStreamRef>> streamsOr =
      captureStreams(extraArgs);

  if (streamsOr.isError())
    return streamsOr.takeError();

  ErrorOr<SmallVector<uint64_t>> normalizedKeysOr =
      normalizeGraphKeysForDevices(graphKeys, streamsOr->size(),
                                   "debug verify replay");
  if (normalizedKeysOr.isError())
    return normalizedKeysOr.takeError();
  ArrayRef<uint64_t> normalizedKeys = *normalizedKeysOr;

  // This validates eager launch traces, not replay safety, so non-capture
  // input signatures are allowed.
  if (ErrorOrSuccess err = validateCapturedPerDeviceKeys(normalizedKeys))
    return err.takeError();

  SmallVector<Driver::AsyncDeviceGraphRef> selectedGraphs;
  selectedGraphs.reserve(normalizedKeys.size());
  for (size_t deviceIdx = 0; deviceIdx < normalizedKeys.size(); ++deviceIdx) {
    const auto &deviceMap = perDeviceGraphs[deviceIdx];
    auto it = deviceMap.find(normalizedKeys[deviceIdx]);
    ASSERT_STREAM(it != deviceMap.end(), "expected captured graph entry");

    selectedGraphs.push_back(it->second.graph.copy());
  }

  Driver::DeviceGraphCollection graphs(std::move(selectedGraphs));
  return graphs.debugVerifyReplay([&]() -> ErrorOrSuccess {
    auto result = executeWithInputs(inputs);
    if (result.isError())
      return result.takeError();
    return success();
  });
}

ErrorOrSuccess ExecutableModel::awaitDeviceGraphs() {
  for (auto &deviceMap : perDeviceGraphs) {
    for (auto &[key, entry] : deviceMap) {
      if (ErrorOrSuccess err = AsyncRT::awaitOrError(entry.graph))
        return err.takeError();
    }
  }
  return success();
}

ErrorOrSuccess
ExecutableModel::releaseCapturedGraphs(ArrayRef<uint64_t> graphKeys) {
  // Releasing on a model with no captures yet is a no-op for every key.
  if (perDeviceGraphs.empty())
    return success();

  // Count devices from the live capture streams, not perDeviceGraphs.size(),
  // so key-count errors match capture/replay/debugVerifyReplay.
  ErrorOr<SmallVector<Driver::DeviceStreamRef>> streamsOr =
      captureStreams(extraArgs);
  if (streamsOr.isError())
    return streamsOr.takeError();

  ErrorOr<SmallVector<uint64_t>> normalizedKeysOr =
      normalizeGraphKeysForDevices(graphKeys, streamsOr->size(), "release");
  if (normalizedKeysOr.isError())
    return normalizedKeysOr.takeError();
  ArrayRef<uint64_t> normalizedKeys = *normalizedKeysOr;

  for (size_t deviceIdx = 0; deviceIdx < normalizedKeys.size(); ++deviceIdx)
    perDeviceGraphs[deviceIdx].erase(normalizedKeys[deviceIdx]);

  return success();
}

ErrorOrSuccess ExecutableModel::printValue(raw_ostream &os,
                                           AnyAsyncValueRef const &value,
                                           ResultOutputStyle style,
                                           unsigned precision) const {
  return GML::printGenericValue(os, value, style, precision);
}

ErrorOrSuccess
ExecutableModel::processResultForBinaryOutput(const AnyAsyncValueRef &result,
                                              const std::string &outputName,
                                              ProcessedResults &results) const {
  if (auto *tensor = dyn_cast<Tensor>(result)) {
    results.insertResultInfo(
        std::make_shared<TensorResultInfo>(outputName, *tensor));

    const std::string illegalChars = "\\/:?\"<>|";
    std::string sanitizedName = outputName;
    std::replace_if(
        sanitizedName.begin(), sanitizedName.end(),
        [&](char c) { return illegalChars.find(c) != std::string::npos; }, '_');

    results.insertTensor(sanitizedName, tensor);
  } else {
    ErrorOr<AsyncRT::AnyAsyncValueRef> outOrError =
        M::getResultForBinaryOutput(result, results);
    if (outOrError.isError())
      results.insertResultInfo(
          std::make_shared<TensorResultInfo>(outOrError.getError(), ""));
    else
      results.insertResultInfo(std::make_shared<M::VariantResultInfo>(
          outputName, outOrError.takeValue()));
  }
  return success();
}

ErrorOr<ModelList> M::loadMGPMEFFile(RCRef<MEFFile> mefFile) {
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  registerMGPPrimitives(primitives);
  primitives.finalize();

  if (auto err = mefFile->installPrimitives(primitives))
    return err.takeError();

  ModelList models;
  auto *moduleInv = mefFile->getTopLevelInvokable();
  if (auto err = moduleInv->walkPrimitives<Model>(
          [&](auto &&m) { models.push_back(std::forward<decltype(m)>(m)); }))
    return err.takeError();

  return std::move(models);
}
