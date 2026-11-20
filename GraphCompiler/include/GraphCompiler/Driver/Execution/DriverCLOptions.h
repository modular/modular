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

#ifndef GENERICML_EXECUTIONDRIVER_DRIVERCLOPTIONS_H
#define GENERICML_EXECUTIONDRIVER_DRIVERCLOPTIONS_H

#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "GraphCompiler/Driver/Execution/ExecutionDriver.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "Support/CommonCLOptions.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/Process.h"

#include <filesystem>
#include <variant>

namespace llvm {
class MemoryBuffer;
}

namespace M {

class TensorSpecParser : public llvm::cl::parser<TensorSpec> {
public:
  using llvm::cl::parser<TensorSpec>::parser;

  bool parse(llvm::cl::Option &options, StringRef argName, StringRef argValue,
             TensorSpec &result);
};
class TensorShapeParser : public llvm::cl::parser<TensorShape> {
public:
  using llvm::cl::parser<TensorShape>::parser;

  bool parse(llvm::cl::Option &options, StringRef argName, StringRef argValue,
             TensorShape &result);
};

struct InputDataSpec {
  size_t tensorIdx;
  std::variant<InputGenerationKind, std::filesystem::path> inputGen;
  TensorGeneratorFlag genFlags;
};

class InputDataParser : public llvm::cl::parser<InputDataSpec> {
public:
  using llvm::cl::parser<InputDataSpec>::parser;

  bool parse(llvm::cl::Option &options, StringRef argName, StringRef argValue,
             InputDataSpec &result) {
    if (auto errOr = parseImpl(options, argName, argValue, result)) {
      llvm::errs() << errOr.getError() << "\n";
      return true;
    }
    return false;
  }

private:
  // Cleaner error handling; parse() wraps it.
  ErrorOrSuccess parseImpl(llvm::cl::Option &options, StringRef argName,
                           StringRef argValue, InputDataSpec &result);
};

/// Core *-executor options, shared with other executables.
class ExecutorDriverOptions : public AsyncRT::CPUDeviceOptions,
                              public CommonOptions {

public:
  enum class OnFailure {
    kContinue,
    kExit,
  };

  /// Warmup runs; only honored when "num-runs" is greater than 1.
  unsigned numWarmup{0};

  unsigned numRuns{1};

  InputGenerationKind inputGenKind{InputGenerationKind::kZeros};

  SmallVector<TensorSpec> inputSpecs{};

  SmallVector<TensorShape> inputShapes{};

  SmallVector<InputDataSpec> inputData{};

  size_t minimumCpuAlignment{64};

  ResultOutputStyle resultOutputStyle{ResultOutputStyle::kCompact};

  unsigned printPrecision{6};

  std::string binaryOutputDir{""};

  /// If set, save every non-warmup run's execution time to this file.
  std::string saveTimings{""};

  OnFailure onFailure{OnFailure::kExit};

  /// Returns whether an executor should stop when a model returns an error.
  bool stopOnFirstError() const { return onFailure == OnFailure::kExit; }
};

/// Core *-executor CLI options, shared with other executables.
class ExecutorDriverCLOptions : public AsyncRT::RuntimeCLOptions,
                                public CommonCLOptions {
public:
  ExecutorDriverOptions &options;
  using CommonCLOptions::CommonCLOptions;
  ExecutorDriverCLOptions(int argc, char **argv, ExecutorDriverOptions &opts,
                          bool skipInitLLVM = false)
      : RuntimeCLOptions(opts), CommonCLOptions(argc, argv, opts, skipInitLLVM),
        options(opts) {}

private:
  llvm::cl::OptionCategory ExecutorDriverOptionsCategory{
      "Executor driver command line options"};
  /// Only honored when "num-runs" is greater than 1.
  M::cl::MOpt<unsigned, true> numWarmup{
      "num-warmup",
      cl::desc("Execute inference model N times in warmup mode (default 0). "
               "This option is only used if \"num-runs\" > 1."),
      llvm::cl::location(options.numWarmup),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<unsigned, true> numRuns{
      "num-runs", cl::desc("Execute inference model N times (default 1)."),
      llvm::cl::location(options.numRuns),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<InputGenerationKind, true> inputGenKind{
      "model-inputs", cl::desc("Specify input tensor generation:"),
      cl::values(
          clEnumValN(
              InputGenerationKind::kRandom,
              stringifyEnum(InputGenerationKind::kRandom),
              "Input tensors are random (distribution is type dependent)"),
          clEnumValN(InputGenerationKind::kZeros,
                     stringifyEnum(InputGenerationKind::kZeros),
                     "All input tensors are zeros"),
          clEnumValN(InputGenerationKind::kNegOnes,
                     stringifyEnum(InputGenerationKind::kNegOnes),
                     "All input tensors are negatives ones"),
          clEnumValN(InputGenerationKind::kOnes,
                     stringifyEnum(InputGenerationKind::kOnes),
                     "All input tensors are ones")),
      llvm::cl::location(options.inputGenKind),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MListOpt<TensorSpec, SmallVector<TensorSpec>, TensorSpecParser>
      inputSpecs{
          "input-specs",
          cl::desc("Specify input specifications. Required when the model has "
                   "no or partially fixed shapes or data types. Input shapes "
                   "('--input-shapes') can be provided additionally, but must "
                   "match the"
                   "model's input specs."),
          llvm::cl::location(options.inputSpecs),
          llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MListOpt<TensorShape, SmallVector<TensorShape>, TensorShapeParser>
      inputShapes{
          "input-shapes",
          cl::desc(
              "Specify input shapes. Required when the model has "
              "dynamic input shapes. If specified and the model has static "
              "input shapes, it must match the model's input shapes. An "
              "empty value means scalar."),
          llvm::cl::location(options.inputShapes),
          llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MListOpt<InputDataSpec, SmallVector<InputDataSpec>, InputDataParser>
      inputData{
          "input-data",
          cl::desc(
              "Specify custom data to use per input. It is specified as "
              "--input-data=tensor_idx::input_source::input_source_attr_list "
              "where:\n"
              "- tensor_idx: int representing the input tensor index\n"
              "- input_source: one of [random|zeros|negones|ones] or "
              "the name of a file containing raw binary tensor data stored in "
              "row-major little-endian order\n"
              "- input_source_attr_list: "
              "optional (default empty) comma separated list of "
              "[Truncate|Cycle] which specify relaxations on use of data in "
              "the input file. If empty, the file size must be an exact "
              "multiple of the input tensor size and every tensor in the input "
              "file may only be used once. If Cycle is specified, there is no "
              "limit on the number of times a tensor in the input file can be "
              "used. If Truncate is specified, the file size need not be an "
              "exact multiple of the tensor size (the remainder will be "
              "ignored)."),
          llvm::cl::location(options.inputData),
          llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<size_t, true> minimumCpuAlignment{
      "minimum-cpu-alignment",
      cl::desc(
          "The minimum byte alignment for all CPU-allocated buffers. Default "
          "is 64 bytes for compatibility with the stock TensorFlow cpuDevice."),
      llvm::cl::location(options.minimumCpuAlignment),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<ResultOutputStyle, true> resultOutputStyle{
      "result-output-style", cl::desc("Specify how to display outputs"),
      cl::values(
          clEnumValN(ResultOutputStyle::kFull, "full",
                     "Display all the elements of result"),
          clEnumValN(
              ResultOutputStyle::kCompact, "compact",
              "Display a few elements at the start and end of each dimension"),
          clEnumValN(
              ResultOutputStyle::kBinary, "binary",
              "Save outputs to binary files. This creates a directory "
              "with one binary file for each output tensor and a YAML file "
              "containing the name and shape of each output tensor. The "
              "output directory containing these files must be "
              "specified with --binary-output-dir=<output_dir>."),
          clEnumValN(ResultOutputStyle::kNone, "none",
                     "Do not display any result details")),
      llvm::cl::location(options.resultOutputStyle),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<unsigned, true> printPrecision{
      "print-precision",
      cl::desc("Specify precision of floating-point outputs (full-mode only)"),
      llvm::cl::location(options.printPrecision),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<std::string, true> binaryOutputDir{
      "binary-output-dir",
      cl::desc(
          "If --result-output-style=binary, specify the output directory."),
      llvm::cl::location(options.binaryOutputDir),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  M::cl::MOpt<std::string, true> saveTimings{
      "save-timings",
      cl::desc("Optional filename to capture the execution latency (in "
               "nanonseconds) per run. Use '-' to output to stdout. Does "
               "not include warmup runs. Ignored in --benchmark mode."),
      llvm::cl::location(options.saveTimings),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};

  /// E.g. `continue` for diagnostic verification.
  M::cl::MOpt<ExecutorDriverOptions::OnFailure, true> onFailure{
      "on-failure",
      llvm::cl::desc("Behavior in case an executed function returns with an "
                     "error. Ignored if there is only one function executed."),
      llvm::cl::values(clEnumValN(ExecutorDriverOptions::OnFailure::kContinue,
                                  "continue",
                                  "Continue executing remaining functions"),
                       clEnumValN(ExecutorDriverOptions::OnFailure::kExit,
                                  "exit", "Stop on first error")),
      llvm::cl::location(options.onFailure),
      llvm::cl::cat(ExecutorDriverOptionsCategory)};
};

/// The driver action; later commands do more processing of the input.
enum class DriverCommand {
  inspect,       // Print information about the input without processing it.
  printMetadata, // Print model information after conversion to MEF in YAML
                 // format. This command is used by other tools to introspect
                 // the model metadata.
  printConstantMetadata, // Print information on all the constants contained
                         // in YAML format.
  compile,               // Compile the input into a MEF file for execution.
  execute,               // Execute the input file.
  getVersion,    // Print the version string from the framework and exit.
  frameworkHelp, // Print the framework's help string.
};

/// Exposes the `DriverCommand` enum with some additional options.
class DriverCommandOptions {

public:
  DriverCommand command{};

  /// Return true if the selected output format is binary and shouldn't be
  /// dumped to the console.
  bool commandHasBinaryOutput() const {
    return command == DriverCommand::compile;
  }

  /// Return true if the command passed in requires an input file.
  bool commandTakesInput() const {
    return command != DriverCommand::getVersion &&
           command != DriverCommand::frameworkHelp;
  }
};
class DriverCommandCLOptions {
private:
  llvm::cl::OptionCategory DriverCommandOptionsCategory{
      "Common driver commands"};

public:
  DriverCommandOptions &options;
  DriverCommandCLOptions(DriverCommandOptions &opts) : options(opts) {}
  M::cl::MOpt<DriverCommand, true> command{
      cl::desc("Command to execute"),
      cl::values(
          clEnumValN(DriverCommand::inspect, "inspect",
                     "Print information about the input without processing it"),
          clEnumValN(DriverCommand::compile, "compile",
                     "Compile the input into a MEF file for execution"),
          clEnumValN(DriverCommand::execute, "execute",
                     "Execute the input file"),
          clEnumValN(DriverCommand::getVersion, "get-version",
                     "Get the version of the loaded framework."),
          clEnumValN(DriverCommand::frameworkHelp, "framework-help",
                     "Get the framework help string.")),
      llvm::cl::Required, llvm::cl::cat(DriverCommandOptionsCategory),
      llvm::cl::location(options.command)};
};

/// Driver options: input format plus ExecutorDriverOptions.
class DriverOptions : public DriverInputFormatOptions,
                      public ExecutorDriverOptions {
public:
  //===--------------------------------------------------------------------===//
  // MEF Emission Configuration
  //===--------------------------------------------------------------------===//

  bool enableMEFMetadata{false};
};
class DriverCLOptions : public DriverInputFormatCLOptions,
                        public ExecutorDriverCLOptions {
public:
  using ExecutorDriverCLOptions::ExecutorDriverCLOptions;
  DriverOptions &options;
  DriverCLOptions(int argc, char **argv, DriverOptions &opts,
                  bool skipInitLLVM = false)
      : DriverInputFormatCLOptions(opts),
        ExecutorDriverCLOptions(argc, argv, opts, skipInitLLVM), options(opts) {
  }
  //===--------------------------------------------------------------------===//
  // MEF Emission Configuration
  //===--------------------------------------------------------------------===//

  M::cl::MOpt<bool, true> enableMEFMetadata{
      "enable-mef-metadata",
      cl::desc("Enable metadata sections for better MEF2MLIR"),
      llvm::cl::location(options.enableMEFMetadata)};
};

/// tfl-graph-verifier tolerances for comparing TFLP and FlatBuffer outputs.
class VerifyGraphDriverOptions : public ExecutorDriverOptions {

public:
  double atol{1.0E-5};
  double rtol{1.0E-8};
};
class VerifyGraphDriverCLOptions : public ExecutorDriverCLOptions {

public:
  VerifyGraphDriverOptions &options;
  VerifyGraphDriverCLOptions(int argc, char **argv,
                             VerifyGraphDriverOptions &opts,
                             bool skipInitLLVM = false)
      : ExecutorDriverCLOptions(argc, argv, opts, skipInitLLVM), options(opts) {
  }

private:
  llvm::cl::OptionCategory VerifyGraphDriverCLOptionsCategory{
      "Graph verification options"};

  M::cl::MOpt<double, true> atol{
      "atol",
      cl::desc("Absolute tolerance for determining whether there's an accuracy "
               "discrepancy between TFLP and stock-tflite."),
      llvm::cl::location(options.atol),
      llvm::cl::cat(VerifyGraphDriverCLOptionsCategory)};

  M::cl::MOpt<double, true> rtol{
      "rtol",
      cl::desc("Relative tolerance for determining whether there's an "
               "accuracy discrepancy between TFLP and stock-tflite."),
      llvm::cl::location(options.rtol),
      llvm::cl::cat(VerifyGraphDriverCLOptionsCategory)};
};

} // namespace M

#endif // GENERICML_EXECUTIONDRIVER_DRIVERCLOPTIONS_H
