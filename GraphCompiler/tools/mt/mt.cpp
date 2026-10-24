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

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Utils.h"
#include "Config/Version.h"
#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "GraphCompiler/Driver/Common/Utils.h"
#include "GraphCompiler/Driver/Execution/DriverCLOptions.h"
#include "GraphCompiler/FrameworkFrontend.h"
#include "GraphCompiler/MEFModel/MEFModel.h"
#include "GraphCompiler/Passes.h"
#include "Init/Init.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/ML/DebugPrint.h"
#include "Support/ML/SizeUtils.h"
#include "Support/Process.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using llvm::MemoryBuffer;

namespace {

//===--------------------------------------------------------------------===//
// GenericDriverCLOptions
//===--------------------------------------------------------------------===//

class GenericDriverOptions : public DriverCommandOptions, public DriverOptions {
public:
  //===--------------------------------------------------------------------===//
  // Arguments forwarded to the framework.
  //===--------------------------------------------------------------------===//

  llvm::SmallVector<std::string> forceUnparsed{};
  llvm::SmallVector<std::string> frameworkArgs{};
};

class GenericDriverCLOptions : public DriverCommandCLOptions,
                               public DriverCLOptions {

public:
  GenericDriverOptions &options;
  GenericDriverCLOptions(int argc, char **argv, GenericDriverOptions &opts,
                         bool skipInitLLVM = false)
      : DriverCommandCLOptions(opts),
        DriverCLOptions(argc, argv, opts, skipInitLLVM), options(opts) {}

private:
  llvm::cl::OptionCategory GenericDriverCLOptionsCategory{
      "Generic driver options"};
  //===--------------------------------------------------------------------===//
  // Arguments forwarded to the framework.
  //===--------------------------------------------------------------------===//

  M::cl::MListOpt<std::string, llvm::SmallVector<std::string>> forceUnparsed{
      "X", cl::desc("Arguments to be passed to the framework."),
      llvm::cl::cat(GenericDriverCLOptionsCategory),
      llvm::cl::location(options.forceUnparsed)};
};
} // namespace

static int setupMLIRAndRun(
    ContextRef context, const ExecutorDriverOptions &options,
    const char *testName, std::unique_ptr<llvm::MemoryBuffer> &&file,
    std::function<LogicalResult(CPUDevice &, MLIRContext *)> &&bodyFn) {
  if (failed(options.configureMLIRContextAndExecute(
          std::move(file), [&](MLIRContext *ctx) {
            AsyncRT::CPUDevice *cpuDevice = context->get<CPUDevice>();
            assert(cpuDevice && "caller should have set up cpuDevice for this");
            registerContext(*ctx, context);
            return AsyncRT::runWithLeakCheckedRuntime(
                testName, [&]() { return bodyFn(*cpuDevice, ctx); });
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

static int runBinaryFile(ContextRef maxCtx, MOFContext &ctx,
                         M::BufferRef fileBuffer, const ModelList &models) {
  const ExecutorDriverOptions *driverOptions = ctx.getExecutorDriverOptions();
  assert(driverOptions && "cannot run a binary file via the API");

  auto runModels =
      [&, context = maxCtx.copy()](CPUDevice &cpuDevice,
                                   MLIRContext *mlirCtx) -> LogicalResult {
    context->emplaceIfMissing<DebugTensorPrintOptions>(
        driverOptions->resultOutputStyle, driverOptions->printPrecision,
        driverOptions->binaryOutputDir);
    bool hadFailure = false;
    for (const auto &model : models) {
      if (failed(runModel(model.get(), cpuDevice, *driverOptions))) {
        hadFailure = true;
        if (driverOptions->stopOnFirstError())
          return failure();
      }
    }
    return failure(hadFailure);
  };

  std::unique_ptr<llvm::MemoryBuffer> aliasedBuffer =
      llvm::MemoryBuffer::getMemBuffer(fileBuffer->getMemBufferRef(),
                                       /*RequiresNullTerminator=*/
                                       false);

  return setupMLIRAndRun(maxCtx.copy(), *driverOptions, "Modular model",
                         std::move(aliasedBuffer), runModels);
}

namespace {
/// Collect the options this binary does not own, and write the rest to \p kept.
/// The framework registers its own options when its dylib loads, which happens
/// well after ParseCommandLineOptions would have rejected them as unknown.
/// argv itself is left alone: MOFContext parses the full command line too.
llvm::SmallVector<std::string>
extractFrameworkArgs(int argc, char **argv,
                     llvm::SmallVectorImpl<const char *> &kept) {
  const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &known =
      llvm::cl::getRegisteredOptions();
  llvm::SmallVector<std::string> extracted;
  kept.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    llvm::StringRef name = arg;
    // A bare "-" is stdin and "--" ends option parsing; neither is an option.
    if (arg.size() > 1 && arg != "--" &&
        (name.consume_front("--") || name.consume_front("-")) &&
        !known.contains(name.split('=').first)) {
      extracted.push_back(arg.str());
      continue;
    }
    kept.push_back(argv[i]);
  }
  return extracted;
}
} // namespace

int main(int argc, char **argv) {
  // Skip InitLLVM's signal handler; Init::Init registers ours.
  GenericDriverOptions options;
  GenericDriverCLOptions parser(argc, argv, options, /*skipInitLLVM=*/true);
  llvm::SmallVector<const char *> parseArgv;
  options.frameworkArgs = extractFrameworkArgs(argc, argv, parseArgv);
  llvm::cl::ParseCommandLineOptions(parseArgv.size(), parseArgv.data(),
                                    "Modular Tool (mt)\n");

  struct InputInfo {
    M::BufferRef file;
    FileFormat format = FileFormat::kUnknown;
    std::filesystem::path inputPath;
  };

  std::vector<InputInfo> inputsInfo{};

  // stdin gets a default-constructed `InputInfo`.
  auto fromStdin = options.inputFilename == "-";
  if (fromStdin) {
    inputsInfo.emplace_back();
  }

  if (options.commandTakesInput() || !fromStdin) {
    auto inputAdaptorRange = llvm::split(options.inputFilename, ",");
    if (std::distance(inputAdaptorRange.begin(), inputAdaptorRange.end()) > 1 &&
        options.command != DriverCommand::execute) {
      return options.reportError(
          "Only the execute command supports multiple inputs right now.");
    }
    for (StringRef inputFilename : inputAdaptorRange) {
      InputInfo info{};
      info.inputPath = std::filesystem::path{inputFilename.str()};

      std::error_code errorCode;
      if (std::filesystem::is_directory(info.inputPath, errorCode))
        return options.reportError("unable to detect input model format");
      if (errorCode) {
        return options.reportError(
            Twine("failed to check type of path: ") + inputFilename +
            " (error code: " + Twine(errorCode.message()) + ")");
      }

      // Aligned to the maximum attribute alignment in case it's MEF.
      std::unique_ptr<llvm::MemoryBuffer> inputFile =
          options.openInputFileOrExit(inputFilename, maxAttributeAlignment());
      info.file = Buffer::take(std::move(inputFile));
      info.format = options.getInputFormat(info.inputPath.extension().string(),
                                           info.file->getBuffer());

      inputsInfo.push_back(std::move(info));
    }
  }

  // The final scenario's return code wins.
  int rc = EXIT_SUCCESS;

  Init::Options contextOptions;
  contextOptions.withCPUDeviceOptions(options);
  auto contextOr =
      Init::createContext(options.getProgramName(), contextOptions);
  if (contextOr.isError())
    return options.reportError(contextOr.getError());

  if (options.command == DriverCommand::getVersion) {
    // llvm::formatv does not support octal.
    char profilingLevelStr[16];
    snprintf(profilingLevelStr, sizeof(profilingLevelStr), "%#o",
             MODULAR_ASYNCRT_MAX_PROFILING_LEVEL);
    llvm::outs() << "mt: " << getModularVersionString()
                 << "\nmt profiling level: " << profilingLevelStr << "\n";
    return EXIT_SUCCESS;
  }

  MOFContext frameworkCtx(contextOr->copy(), argc, argv);

  if (options.command == DriverCommand::frameworkHelp) {
    if (auto err = frameworkCtx.setOptions({"-h"}))
      return options.reportError(err.getError());
    return EXIT_SUCCESS;
  }

  SmallVector<std::string> dylibOptions;

  dylibOptions.append(options.forceUnparsed.begin(),
                      options.forceUnparsed.end());
  dylibOptions.append(options.frameworkArgs.begin(),
                      options.frameworkArgs.end());

  if (options.saveTemps) {
    dylibOptions.push_back("--save-temps");
    dylibOptions.push_back("--temps-dir=" + options.tempsDir);
  }

  if (auto err = frameworkCtx.setOptions(dylibOptions))
    return options.reportError(err.getError());
  frameworkCtx.setDriverOptions(options);

  for (auto &inputInfo : inputsInfo) {

    // Inspect works on any input without transformation.
    if (options.command == DriverCommand::inspect) {
      std::unique_ptr<llvm::ToolOutputFile> outputFile =
          options.getOutputFile(options.commandHasBinaryOutput());
      auto identityOr =
          getFileContentIdentity(inputInfo.file.copy(), inputInfo.format);
      if (failed(identityOr))
        return options.reportError(identityOr.getError());

      outputFile->os() << *identityOr << ".\n";
      outputFile->keep();
      return EXIT_SUCCESS;
    }

    if (options.command == DriverCommand::compile) {
      auto outputFile = options.getOutputFile(options.commandHasBinaryOutput());

      if (inputInfo.format == FileFormat::kMEFFile) {
        outputFile->os().write(inputInfo.file->getBufferStart(),
                               inputInfo.file->getBufferSize());
        outputFile->keep();
        return EXIT_SUCCESS;
      }

      auto bufferOr =
          frameworkCtx.compileToBinary(inputInfo.file.copy(), inputInfo.format);
      if (bufferOr.isError())
        return options.reportError(bufferOr.getError());

      M::BufferRef buffer = bufferOr.takeValue();
      outputFile->os().write(buffer->getBufferStart(), buffer->getBufferSize());
      outputFile->keep();
      return EXIT_SUCCESS;
    }

    M::BufferRef buffer;

    if (inputInfo.format != FileFormat::kMEFFile) {
      auto bufferOr =
          frameworkCtx.compileToBinary(inputInfo.file.copy(), inputInfo.format);
      if (bufferOr.isError())
        return options.reportError(bufferOr.getError());

      inputInfo.file = bufferOr.takeValue();
    }

    if (!(options.command == DriverCommand::printMetadata ||
          options.command == DriverCommand::printConstantMetadata ||
          options.command == DriverCommand::execute)) {
      return options.reportError("Unknown command:");
    }

    auto mefFileOr = MEFFile::open(inputInfo.file->getBuffer());
    if (failed(mefFileOr))
      return options.reportError(mefFileOr.getError());
    auto modelsOr = M::loadMGPMEFFile(*mefFileOr);
    if (modelsOr.isError())
      return options.reportError(modelsOr.getError());

    rc = runBinaryFile(contextOr->copy(), frameworkCtx, inputInfo.file.copy(),
                       *modelsOr);
  }

  return rc;
}
