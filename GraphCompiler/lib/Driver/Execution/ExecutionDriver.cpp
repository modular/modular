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

#include "GraphCompiler/Driver/Execution/ExecutionDriver.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "GraphCompiler/Driver/Device/DeviceMap.h"
#include "GraphCompiler/Driver/Execution/DriverCLOptions.h"
#include "GraphCompiler/Driver/Execution/DriverInternals.h"
#include "GraphCompiler/MEFModel/MEFModel.h"
#include "GraphCompiler/Support/FormatYAML.h"
#include "GraphCompiler/Support/ResultInfo.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/Error.h"
#include "Support/ML/SizeUtils.h"
#include "Support/Profiling/TimeProfiler.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using mlir::MLIRContext;

StringLiteral M::stringifyEnum(InputGenerationKind kind) {
  switch (kind) {
  case InputGenerationKind::kRandom:
    return "random";
  case InputGenerationKind::kZeros:
    return "zeros";
  case InputGenerationKind::kOnes:
    return "ones";
  case InputGenerationKind::kNegOnes:
    return "negones";
  }
  llvm_unreachable("unhandled InputGenerationKind");
}

std::optional<InputGenerationKind>
M::symbolizeInputGenerationKind(StringRef s) {
  return llvm::StringSwitch<std::optional<InputGenerationKind>>(s)
      .Case(stringifyEnum(InputGenerationKind::kRandom),
            InputGenerationKind::kRandom)
      .Case(stringifyEnum(InputGenerationKind::kZeros),
            InputGenerationKind::kZeros)
      .Case(stringifyEnum(InputGenerationKind::kOnes),
            InputGenerationKind::kOnes)
      .Case(stringifyEnum(InputGenerationKind::kNegOnes),
            InputGenerationKind::kNegOnes)
      .Default(std::nullopt);
}

std::optional<TensorGeneratorFlag>
M::symbolizeTensorGeneratorFlag(StringRef s) {
  return llvm::StringSwitch<std::optional<TensorGeneratorFlag>>(s)
      .Case("Cycle", TensorGeneratorFlag::kCycle)
      .Case("Truncate", TensorGeneratorFlag::kTruncate)
      .Default(std::nullopt);
}

bool TensorShapeParser::parse(llvm::cl::Option &options, StringRef argName,
                              StringRef argValue, TensorShape &result) {
  ErrorOr<TensorShape> errorOrTensorShape =
      TensorShape::parseFromString(argValue);
  if (failed(errorOrTensorShape))
    return options.error(errorOrTensorShape.getError());
  result = *errorOrTensorShape;
  return false;
}

bool TensorSpecParser::parse(llvm::cl::Option &options, StringRef argName,
                             StringRef argValue, TensorSpec &result) {
  ErrorOr<TensorSpec> errorOrTensorSpec = TensorSpec::parseFromString(argValue);
  if (failed(errorOrTensorSpec))
    return options.error(errorOrTensorSpec.getError());
  result = *errorOrTensorSpec;
  return false;
}

// Parses --input-data=tensor_idx::input_source::input_source_attr_list.
//    - tensor_idx: input tensor index
//    - input_source: [zeros|random|ones|negones] (an InputGenerationKind) or a
//      filename
//    - input_source_attr_list (optional): csv of [Cycle|Truncate]
//      (TensorGeneratorFlags)
ErrorOrSuccess InputDataParser::parseImpl(llvm::cl::Option &options,
                                          StringRef argName, StringRef argValue,
                                          InputDataSpec &result) {
  llvm::SmallVector<llvm::StringRef, 3> args;
  argValue.split(args, "::");
  if (args.size() < 2)
    return Error("--input-data requires `::` separated list of args.");
  if (args.size() > 3)
    return Error("--input-data supports max of 3 args separated by `::`.");

  auto tensorIdxStr = args[0];
  auto err = tensorIdxStr.getAsInteger(10, result.tensorIdx);
  if (err)
    return Error(Twine("Could not convert tensorIdx (") + tensorIdxStr +
                 ") to integer.");

  auto filenameOrGenKindStr = args[1];
  std::optional<InputGenerationKind> genKind =
      symbolizeInputGenerationKind(filenameOrGenKindStr.str());
  if (genKind.has_value()) {
    result.inputGen = genKind.value();
    if (args.size() > 2)
      return Error(Twine("No additional --input-data flags (") + args[2] +
                   ") are supported when input_source is an input "
                   "generation function.");
    return success();
  }
  auto filename = std::filesystem::path(filenameOrGenKindStr.str());
  if (!std::filesystem::exists(filename)) {
    return Error(Twine("--input-data file (") + filename.string() +
                 ") does not exist.");
  }
  result.inputGen = filename;

  if (args.size() < 3)
    return success();
  for (const auto genFlagStr : llvm::split(args[2], ",")) {
    auto flag = symbolizeTensorGeneratorFlag(genFlagStr);
    if (!flag)
      return Error(Twine("Cannot convert ") + genFlagStr +
                   "to TensorGeneratorFlag.");
    result.genFlags = result.genFlags | flag.value();
  }

  return success();
}

/// Replace non-output-conforming results with their equivalents (such as
/// `Tensor`).
static void
convertResultsToGenericValues(const ExecutableModel *model,
                              MutableArrayRef<AnyAsyncValueRef> results) {
  for (auto &result : results) {
    ASSERT_STREAM(result.isReady(), << "tensor result should be ready");
    if (result.isError() || result.isType<Tensor>())
      continue;
    await(result);
  }
}

/// Waits for results and returns the first error, if any.
static ErrorOrSuccess awaitAndCheck(MutableArrayRef<AnyAsyncValueRef> results) {
  await(results);
  for (const auto &result : results) {
    if (auto *diag = result.getDiagnosticIfPresent())
      // TODO: Capture location
      return std::move(diag->getMessage());
  }
  return success();
}

static LogicalResult printDiagIfError(const AnyAsyncValueRef &res) {
  if (auto *diag = res.getDiagnosticIfPresent()) {
    llvm::outs() << "<error: " << diag->getMessage() << '>' << "\n";
    auto loc = diag->decodeLocation();
    llvm::outs() << "loc: " << loc.filename << ':' << loc.line << ':'
                 << loc.column << "\n";
    return failure();
  }
  return success();
}

static LogicalResult
writeYamlToFile(const std::filesystem::path &yamlFilePath,
                SmallVector<std::shared_ptr<ResultInfo>> &tensorResultsInfo) {
  std::error_code yamlFileErr;
  llvm::raw_fd_ostream yamlFile(llvm::StringRef(yamlFilePath.string()),
                                yamlFileErr);
  if (yamlFileErr) {
    llvm::errs() << "Failed to open output tensor yaml: "
                 << yamlFileErr.message() << "\n";
    return failure();
  }

  llvm::yaml::Output yout(yamlFile);
  yout << tensorResultsInfo;

  return success();
}

/// Prints model results in the executor tool's standard format, reporting
/// errors compatibly with diagnostic verification. The leading chain result is
/// never printed, but its error is reported.
///
/// With --result-output-style=binary, instead writes one binary file per tensor
/// result plus a YAML of names and shapes to --binary-output-dir.
static LogicalResult outputResults(const ExecutableModel *model,
                                   MutableArrayRef<AnyAsyncValueRef> results,
                                   ResultOutputStyle outputStyle,
                                   unsigned precision, StringRef outputDir) {
  auto &os = llvm::outs();
  StringRef name = model->getModel()->getName();
  os << "'" << name << "' returned ";

  ASSERT_STREAM(!results.empty(), << "must have at least one result");
  const AnyAsyncValueRef &ch = results.front();
  if (failed(printDiagIfError(ch)))
    return mlir::failure();
  ASSERT_STREAM(ch.isValueAvailable() && isa<Chain>(ch),
                << "first result must be a chain");
  results = results.drop_front();

  if (results.empty()) {
    os << '\n';
    return success();
  }

  convertResultsToGenericValues(model, results);

  [[maybe_unused]] bool allResultsReady =
      std::all_of(results.begin(), results.end(),
                  [](AnyAsyncValueRef &result) { return result.isReady(); });
  ASSERT_STREAM(allResultsReady, << "model result must be ready");

  if (outputStyle == ResultOutputStyle::kBinary) {

    os << "--- Writing Outputs to " << outputDir << "\n";

    if (auto ec =
            llvm::sys::fs::create_directories(outputDir,
                                              /* ignoreExisting */ true)) {
      llvm::errs() << "Failed to create output tensor directories: "
                   << ec.message() << "\n";
      return failure();
    }

    ArrayRef<std::string> outputNames = model->getModel()->getOutputNames();

    ProcessedResults processedResults;

    bool hasError = false;
    for (size_t i = 0; i < results.size(); ++i) {
      const std::string &outputName = outputNames[i];
      const AnyAsyncValueRef &result = results[i];
      if (failed(printDiagIfError(result))) {
        hasError = true;
      } else {
        if (failed(model->processResultForBinaryOutput(result, outputName,
                                                       processedResults)))
          hasError = true;
      }
    }

    for (auto &[name, tensor] : processedResults.getTensors()) {
      std::ofstream outputFile((std::filesystem::path(outputDir.str()) / name),
                               std::ofstream::binary);

      outputFile.write((const char *)tensor->getBuffer(),
                       tensor->getSpec().getSizeInBytes());

      if (outputFile.fail()) {
        llvm::errs() << "Failed to write output tensor file.\n";
        return failure();
      }
    }

    // The model tool uses this to map outputs for verification.
    if (writeYamlToFile(
            (std::filesystem::path(outputDir.str()) / "output.yaml"),
            processedResults.getResultInfos())
            .failed()) {
      llvm::errs() << "Failed to write YAML with result info.\n";
    }
    return failure(hasError);
  }

  bool first = true;
  bool hasError = false;

  // Print in output-name order.
  const auto &outputNames = model->getModel()->getOutputNames();
  size_t numResults = outputNames.size();
  std::vector<size_t> sortedIndices(numResults);
  std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&outputNames](size_t i1, size_t i2) {
              return outputNames[i1] < outputNames[i2];
            });

  for (size_t idx : sortedIndices) {
    const auto &result = results[idx];
    if (!first)
      os << ", ";
    first = false;
    if (failed(printDiagIfError(result)))
      hasError = true;
    else {
      if (auto *tensor = dyn_cast<Tensor>(result)) {
        if (auto err = tensor->print(os, outputStyle, precision)) {
          os << "<error: " << err.getError() << '>';
          hasError = true;
        }
      } else {
        // Not generically printable; defer to the framework model.
        if (auto err = model->printValue(os, result, outputStyle, precision)) {
          os << "<error: " << err.getError() << '>';
          hasError = true;
        }
      }
    }
  }
  os << '\n';
  return failure(hasError);
}

namespace {
/// Driver devices for a model's accelerator-placed inputs and results. Inputs
/// are generated on the host and staged onto their device; results are copied
/// back before printing. `deviceMap` goes to `Model::setup` so the model runs
/// on the same contexts as the staged buffers.
struct ModelDeviceState {
  GraphCompiler::Driver::DeviceMap deviceMap;
  /// Parallel to the model's inputs; null for host-placed inputs.
  SmallVector<M::Driver::DeviceContextRef> inputDevices;
  /// Parallel to the model's tensor and buffer results; null for host-placed
  /// results.
  SmallVector<M::Driver::DeviceContextRef> outputDevices;
  /// Host endpoint of the staging copies; null when every placement is host.
  M::Driver::DeviceContextRef hostDevice;
};
} // namespace

/// Creates a device context for every accelerator the model places an input or
/// result on.
static ErrorOr<ModelDeviceState> createModelDeviceState(const Model *model) {
  ModelDeviceState state;
  auto isHost = [](const DeviceRef &ref) { return ref.label == kCPULabel; };
  if (llvm::all_of(model->getInputDevices(), isHost) &&
      llvm::all_of(model->getOutputDevices(), isHost))
    return state;

  // `state.deviceMap` doubles as the memo for this loop.
  auto materialize = [&](ArrayRef<DeviceRef> placements,
                         SmallVector<M::Driver::DeviceContextRef> &devices,
                         StringRef kind) -> ErrorOrSuccess {
    for (const DeviceRef &ref : placements) {
      if (isHost(ref)) {
        devices.emplace_back();
        continue;
      }
      if (auto it = state.deviceMap.find(ref); it != state.deviceMap.end()) {
        devices.push_back(it->second);
        continue;
      }
      // `getOrCreate` already checks isCompatible().
      ErrorOr<M::Driver::DeviceContextRef> ctxOr =
          M::Driver::DeviceContext::getOrCreate(M::Driver::Device::gpuAPI,
                                                ref.id);
      if (ctxOr.isError())
        return ctxOr.takeError();
      M::Driver::DeviceContextRef ctx = ctxOr.takeValue();

      state.deviceMap.try_emplace(ref, ctx);
      devices.push_back(std::move(ctx));
    }
    return success();
  };

  // Inputs pair positionally with the generated tensors, so staging needs a
  // placement for every input. (Placements skip non-tensor entries, which
  // entry inputs never contain in practice.)
  ArrayRef<DeviceRef> inputPlacements = model->getInputDevices();
  if (!llvm::all_of(inputPlacements, isHost)) {
    if (inputPlacements.size() != model->getInputSpecs().size())
      return Error(llvm::formatv(
          "model '{}' has device-placed inputs, but its placements do not "
          "cover every input, so they cannot be staged"));

    if (auto err = materialize(inputPlacements, state.inputDevices, "input"))
      return err.takeError();
  }

  if (auto err =
          materialize(model->getOutputDevices(), state.outputDevices, "result"))
    return err.takeError();

  ErrorOr<M::Driver::DeviceContextRef> hostOr =
      M::Driver::DeviceContext::create(M::Driver::Device::cpuAPI);
  if (hostOr.isError())
    return Error(Twine("failed to create the host device for staging model "
                       "inputs and results: ") +
                 Twine(hostOr.takeError().get()));
  state.hostDevice = hostOr.takeValue();
  return state;
}

// Returns a ready AsyncValueRef<Tensor> per generator for `execute`. A host
// input borrows the generator's next tensor; a device input is copied to a
// fresh allocation on the model's device context, ordering the copy before the
// model's kernels.
static ErrorOr<SmallVector<AnyAsyncValueRef>>
buildArgs(CPUDevice &cpuDevice, ArrayRef<ITensorGeneratorPtr> generators,
          const ModelDeviceState &deviceState) {
  SmallVector<AnyAsyncValueRef> args;
  args.reserve(generators.size());
  for (const auto &[i, generator] : llvm::enumerate(generators)) {
    const Tensor &generated = generator->getNextTensor();
    M::Driver::DeviceContextRef device = i < deviceState.inputDevices.size()
                                             ? deviceState.inputDevices[i]
                                             : M::Driver::DeviceContextRef();
    if (!device) {
      args.push_back(
          AsyncValueRef<Tensor>::createReady(cpuDevice, generated.borrow()));
      continue;
    }

    ErrorOr<M::Driver::DeviceBufferRef> bufferOr =
        GraphCompiler::Driver::allocateDeviceBuffer(device,
                                                    generated.getSpec());
    if (bufferOr.isError())
      return bufferOr.takeError();
    M::Driver::DeviceBufferRef buffer = bufferOr.takeValue();

    // The generator owns the buffer and outlives the enqueued copy.
    M::Driver::DeviceBufferRef hostView = GraphCompiler::Driver::borrowBuffer(
        deviceState.hostDevice, const_cast<void *>(generated.getBuffer()),
        generated.getSpec().getSizeInBytes());
    if (ErrorOrSuccess err = hostView->copyTo(*buffer))
      return err.takeError();

    args.push_back(GraphCompiler::Driver::prepareInput(cpuDevice, buffer,
                                                       generated.getSpec()));
  }
  return args;
}

static bool isTensorOrBuffer(AnyAsyncValueRef &value) {
  return value.isType<Tensor>() || value.isType<TensorBufferRef>();
}

/// Replaces every device-placed tensor result with a host copy for printing.
/// The copy is enqueued on the model's device context, after its kernels, and
/// the device is drained before the host read.
static ErrorOrSuccess
stageResultsToHost(CPUDevice &cpuDevice, const ModelDeviceState &deviceState,
                   MutableArrayRef<AnyAsyncValueRef> results) {
  // Skip the leading chain, as `outputResults` does.
  if (!results.empty() && isa<Chain>(results.front()))
    results = results.drop_front();

  // Only tensors and buffers consume a placement.
  size_t placementIdx = 0;
  for (AnyAsyncValueRef &result : results) {
    if (placementIdx >= deviceState.outputDevices.size())
      break;
    // An errored result's type is unknown, so the remaining pairing is too;
    // leave the rest to the printer's error reporting.
    if (result.getDiagnosticIfPresent())
      break;

    if (!isTensorOrBuffer(result))
      continue;

    const M::Driver::DeviceContextRef &device =
        deviceState.outputDevices[placementIdx++];
    if (!device)
      continue;

    const M::MLRT::Tensor *tensor = dyn_cast<Tensor>(std::as_const(result));
    // A bare buffer prints as metadata only.
    if (!tensor)
      continue;

    TensorSpec spec = tensor->getSpec();
    // The result value keeps the buffer alive across the copy.
    M::Driver::DeviceBufferRef deviceView = GraphCompiler::Driver::borrowBuffer(
        device, const_cast<void *>(tensor->getBuffer()), spec.getSizeInBytes());

    ErrorOr<M::Driver::DeviceBufferRef> hostOr =
        GraphCompiler::Driver::allocateDeviceBuffer(deviceState.hostDevice,
                                                    spec);
    if (hostOr.isError())
      return hostOr.takeError();

    M::Driver::DeviceBufferRef hostBuffer = hostOr.takeValue();
    if (ErrorOrSuccess err = deviceView->copyTo(*hostBuffer))
      return err.takeError();

    if (ErrorOrSuccess err = device->synchronize())
      return err.takeError();

    result = GraphCompiler::Driver::prepareInput(cpuDevice, hostBuffer, spec);
  }
  return success();
}

/// Executes and awaits the results. Without checkResults, the consumer must
/// check them for errors.
static ErrorOr<SmallVector<AnyAsyncValueRef>> executeModelSynchronously(
    CPUDevice &cpuDevice, ExecutableModel &executableModel,
    MutableArrayRef<AnyAsyncValueRef> args, bool checkResults) {
  auto results = executableModel.execute(args);

  if (checkResults) {
    auto unitOr = awaitAndCheck(results);
    if (unitOr.isError())
      return unitOr.takeError();
  } else {
    await(results);
  }
  return results;
}

/// Evaluates the model numRuns times, printing each run's results.
static ErrorOr<std::unique_ptr<ExecutableModel>>
executeModel(const Model *model, CPUDevice &cpuDevice,
             const ExecutorDriverOptions &options,
             ArrayRef<QuantTensorSpec> inputSpecs, unsigned int numRuns,
             unsigned int numWarmup) {
  std::unique_ptr<llvm::raw_fd_ostream> optTimingOs;
  if (!options.saveTimings.empty()) {
    std::error_code errorCode;
    optTimingOs =
        std::make_unique<llvm::raw_fd_ostream>(options.saveTimings, errorCode);
    if (errorCode)
      return Error(
          Twine("failed to open timings file: " + errorCode.message()));
  }

  TimeTraceScope scope(DriverProfilerEntry::create("executeModel"));

  auto deviceStateOr = createModelDeviceState(model);
  if (deviceStateOr.isError())
    return deviceStateOr.takeError();
  ModelDeviceState deviceState = deviceStateOr.takeValue();

  std::unique_ptr<ExecutableModel> executableModel;
  {
    TimeTraceScope scope(DriverProfilerEntry::create("executeModel.setup"));
    auto executableModel1Or =
        model->setup(cpuDevice, /*weightsRegistry=*/{}, deviceState.deviceMap);
    if (executableModel1Or.isError())
      return executableModel1Or.takeError().getMessage().copy();

    executableModel = executableModel1Or.takeValue();
    if (ErrorOrSuccess error = executableModel->waitForModelSetup())
      return error.takeError();
  }

  // Each generator owns its tensors.
  auto inputTensorGeneratorsOr = createTensorGenerators(
      options.inputGenKind, inputSpecs, options.inputData,
      options.minimumCpuAlignment, cpuDevice);
  if (inputTensorGeneratorsOr.isError())
    return inputTensorGeneratorsOr.takeError();
  auto inputTensorGenerators = std::move(*inputTensorGeneratorsOr);

  if (numWarmup > 0) {
    llvm::outs() << "--- Warming up '" << model->getName().str() << "' "
                 << numWarmup << " times\n";
    llvm::outs().flush();
  }
  for (unsigned i = 0; i < numWarmup; ++i) {
    TimeTraceScope scope(DriverProfilerEntry::create("executeModel.warmup"));
    auto argsOr = buildArgs(cpuDevice, inputTensorGenerators, deviceState);
    if (argsOr.isError())
      return argsOr.takeError();
    auto args = std::move(*argsOr);

    auto resultsOr =
        executeModelSynchronously(cpuDevice, *executableModel, args,
                                  /*checkResults=*/true);
    if (resultsOr.isError())
      return resultsOr.takeError();
  }

  if (numRuns > 0) {
    llvm::outs() << "--- Running '" << model->getName().str() << "'";
    if (numRuns > 1)
      llvm::outs() << " " << numRuns << " times:\n";
    else
      llvm::outs() << ":\n";
    llvm::outs().flush();
  }

  for (unsigned i = 0; i < numRuns; ++i) {
    ErrorOr<SmallVector<AnyAsyncValueRef>> argsOr =
        buildArgs(cpuDevice, inputTensorGenerators, deviceState);
    if (argsOr.isError())
      return argsOr.takeError();

    SmallVector<AnyAsyncValueRef> args = argsOr.takeValue();

    if (i > 0 && i % 100 == 0)
      llvm::outs() << "--- Run " << i << "...\n";

    std::chrono::high_resolution_clock::time_point startTime, endTime;
    auto entry = DriverProfilerEntry::create("executeModel.run");

    startTime = std::chrono::high_resolution_clock::now();
    // When printing, errors are reported by the printer below.
    auto resultsOr = executeModelSynchronously(
        cpuDevice, *executableModel, args,
        /*checkResults=*/options.resultOutputStyle == ResultOutputStyle::kNone);
    if (resultsOr.isError())
      return resultsOr.takeError();
    auto results = std::move(*resultsOr);
    endTime = std::chrono::high_resolution_clock::now();

    std::move(entry).record();

    if (options.resultOutputStyle != ResultOutputStyle::kNone) {
      if (ErrorOrSuccess err =
              stageResultsToHost(cpuDevice, deviceState, results))
        return err.takeError();

      // AsyncValue errors are printed here and a generic error returned, so
      // unit tests can tell expected model errors from tool failures.
      if (failed(outputResults(
              executableModel.get(), results, options.resultOutputStyle,
              options.printPrecision, options.binaryOutputDir)))
        return Error("repeated model execution failed");
    }

    // Bare nanoseconds, for scripts. Failed executions report no time.
    if (optTimingOs) {
      *optTimingOs << (endTime - startTime).count() << "\n";
      // Flush so an interrupted run still has usable data.
      optTimingOs->flush();
    }

    llvm::outs().flush();
  }

  return executableModel;
}

/// Applies run-time override shapes to compile-time QuantTensorSpecs.
static ErrorOr<SmallVector<QuantTensorSpec>>
getResolvedInputShapes(ArrayRef<QuantTensorSpec> compiledShapes,
                       ArrayRef<TensorSpec> userInputShapes) {
  SmallVector<QuantTensorSpec> ret;
  auto numInputTensors = compiledShapes.size();

  if (userInputShapes.empty()) {
    ret.append(compiledShapes.begin(), compiledShapes.end());
    return ret;
  }

  if (userInputShapes.size() != numInputTensors) {
    return Error(Twine("Number of specified input shapes (") +
                 Twine(userInputShapes.size()) + ") doesn't match " +
                 "number of model inputs (" + Twine(numInputTensors) + ")");
  }

  for (unsigned i = 0; i < numInputTensors; ++i) {
    auto userShape = userInputShapes[i];
    auto compiledShape = compiledShapes[i];
    if (auto errOr = compiledShape.isRefinedBy(userShape))
      return errOr.takeError();
    if (compiledShape.getEltType() == DType::invalid)
      compiledShape.setEltType(userShape.getEltType());
    ret.push_back(compiledShape.cloneWithNewShape(userShape));
  }

  return ret;
}

/// Pass resolved input shapes to profiler, if it's active.
static void saveInputSpecs(ArrayRef<QuantTensorSpec> specs,
                           std::optional<TimeTraceProfiler> &profiler) {
  if (profiler)
    for (auto &spec : specs)
      profiler->addInputShape(spec.getAsString());
}

/// Run `model` using the command line options flags configured by the user.
LogicalResult M::runModel(const Model *model, AsyncRT::CPUDevice &cpuDevice,
                          const ExecutorDriverOptions &options) {
  SmallVector<TensorSpec> userInputSpecs;
  if (!options.inputSpecs.empty()) {
    userInputSpecs = SmallVector<TensorSpec>(options.inputSpecs.begin(),
                                             options.inputSpecs.end());
  } else if (!options.inputShapes.empty()) {
    for (auto [modelSpec, userShape] :
         llvm::zip(model->getInputSpecs(), options.inputShapes))
      userInputSpecs.emplace_back(userShape, modelSpec.getEltType());
  }
  auto shapesOrErr =
      getResolvedInputShapes(model->getInputSpecs(), userInputSpecs);
  if (shapesOrErr.isError()) {
    return failure(options.reportError(shapesOrErr.getError()));
  }
  const auto &inputSpecs = shapesOrErr.takeValue();
  saveInputSpecs(inputSpecs, cpuDevice.getProfiler());

  auto executableModelOr = executeModel(model, cpuDevice, options, inputSpecs,
                                        options.numRuns, options.numWarmup);
  if (executableModelOr.isError())
    return failure(options.reportError(executableModelOr.getError()));
  return success();
}
