//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file provides various utilities for configuring and compiling Mojo.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
#define KGEN_TOOLS_MOJO_COMMON_COMPILATION_H

#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/DriverOptions.h"
#include "Mojo/ToolCommon/FrontendOptions.h"
#include "Mojo/ToolCommon/OptionNormalizers.h"
#include "Mojo/ToolCommon/PassManagerConfigOptions.h"
#include "Mojo/ToolCommon/TargetOptions.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/Driver/OptionMarshalling.h"
#include "Support/ErrorOr.h"
#include "mlir/Support/Timing.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class PassManager;
} // namespace mlir

namespace M {
namespace AsyncRT {
class CPUDevice;
struct CPUDeviceOptions;
} // namespace AsyncRT

class TargetInfoAttr;

/// Expands a subcommand's `MOJO_OPTION_WITH_MARSHALLING` rows from `args`
/// into the structs their key paths name, adding invalid values to the errors.
using ExpandMarshalledOptionsFn = llvm::function_ref<void(
    const llvm::opt::InputArgList &args, State &state,
    DriverOptions &driverOptions, KGEN::FrontendOptions &frontendOptions,
    KGEN::CompilationOptions &compilationOptions, TargetArgs &targetArgs,
    MarshallingErrors &errors)>;

/// Result of parsing common Mojo arguments.
struct CommonParseResult {
  /// If set, the caller should exit immediately with this code.
  std::optional<int> exitCode;

  /// The parsed options that the driver acts on itself.
  DriverOptions driverOptions;

  /// The parsed compilation options.
  KGEN::CompilationOptions compilationOptions;

  /// The parsed frontend options.
  KGEN::FrontendOptions frontendOptions;

  /// The parsed target information.
  TargetInfoAttr target;
};

/// Interpret the `args` that `State::parseArguments` parsed for mojo build or
/// mojo run.
///
/// This function extracts the common argument handling shared between
/// the two commands, including:
/// - Diagnostic format parsing
/// - Invalid argument rejection
/// - Input file validation and opening
/// - Source manager setup
/// - Compilation option parsing
/// - Target option parsing
/// - The `--print-*` target queries, which exit without needing an input file
///
/// `expandOptions` reads the subcommand's marshalled options, since only the
/// subcommand sees its option enum.
///
/// Returns a CommonParseResult containing either an exit code (if parsing
/// failed) or the parsed options.
ErrorOr<CommonParseResult>
parseCommonMojoArguments(State &state, const llvm::opt::InputArgList &args,
                         llvm::SourceMgr &sourceManager, MLIRContext &ctx,
                         ExpandMarshalledOptionsFn expandOptions);

/// Set `includePaths` as the include directories of `sourceMgr`, and register
/// the dialects compilation needs on `ctx`.
void initializeCompilation(llvm::ArrayRef<std::string> includePaths,
                           llvm::SourceMgr &sourceMgr, MLIRContext &ctx);

/// Set the warning policy that `state` parsed with the diagnostic format on
/// `options`.
void setWarningPolicy(const State &state, KGEN::FrontendOptions &options);

/// Warn users when doing debug builds with a compiler in debug mode.
void warnBuildingForDebugWithDebugBuiltCompiler(
    const State &state,
    KGEN::CompilationOptions::DebugInfoLevel debugInfoLevel);

/// Where the timing reports of one compiler command go, and in which format.
/// `--timing-json` asks for JSON rather than text, and `--timing-file` names a
/// file rather than stderr; the two compose, so any of the four combinations
/// is reachable.
///
/// A JSON command produces one object for the whole command, with one member
/// per report, so this class owns the braces and the commas between members. A
/// text command leaves each report to write its own titled section.
///
/// The file opens during `configure`, so an unwritable path fails before the
/// compilation runs rather than after it.
class TimingReportSink {
public:
  // The destructor closes the JSON object; a copy would close it twice.
  TimingReportSink() = default;
  TimingReportSink(const TimingReportSink &) = delete;
  TimingReportSink &operator=(const TimingReportSink &) = delete;
  TimingReportSink(TimingReportSink &&) = delete;
  TimingReportSink &operator=(TimingReportSink &&) = delete;

  /// Takes `--timing-json` and opens the file named by `--timing-file`, if the
  /// command gave one.
  ErrorOrSuccess configure(const DriverOptions &driverOptions);

  bool isJSON() const { return json; }

  /// The destination, which is stderr unless `--timing-file` named a file.
  llvm::raw_ostream &stream();

  /// Opens member `name` and returns the stream for its value. JSON only.
  /// Call once per member: it also emits the punctuation around it.
  llvm::raw_ostream &beginMember(StringRef name);

  /// Closes the JSON object and the file. Idempotent.
  void finish();

  ~TimingReportSink() { finish(); }

private:
  /// Null when the reports go to stderr.
  std::unique_ptr<llvm::raw_fd_ostream> file;
  bool json = false;
  bool wroteMember = false;
  bool finished = false;
};

/// This class holds the MLIR timing manager for one compiler command, and
/// gives the root scope that the steps of the compilation nest under. The
/// timing is off unless the `--mlir-timing` option is present. A command
/// without the option nests the parse under an empty scope, which records
/// nothing, and leaves `PassManagerConfigOptions::timingScope` unset, so no
/// pass manager gains timing instrumentation.
class MLIRPassTiming {
public:
  // `passManagerOptions` gives the address of `root` to other code. A move of
  // this object makes that address invalid. A copy is also not correct,
  // because each copy prints the same report again.
  MLIRPassTiming() = default;
  MLIRPassTiming(const MLIRPassTiming &) = delete;
  MLIRPassTiming &operator=(const MLIRPassTiming &) = delete;
  MLIRPassTiming(MLIRPassTiming &&) = delete;
  MLIRPassTiming &operator=(MLIRPassTiming &&) = delete;

  /// Makes the timing active if the `--mlir-timing` option is present. Sets
  /// the display mode from the `--mlir-timing-display` option. Gives an error
  /// if the display mode is not correct.
  /// `sink` takes the report and must outlive this object.
  ErrorOrSuccess configure(const DriverOptions &driverOptions,
                           TimingReportSink &sink);

  /// The root scope for the steps of the compilation. The scope is empty if
  /// the timing is off.
  mlir::TimingScope &rootScope() { return root; }

  /// The pass manager configuration that sends the times to this scope. The
  /// scope stays empty if the timing is off. Then a build that has the trace
  /// function keeps its own timing manager.
  KGEN::PassManagerConfigOptions passManagerOptions();

  /// Prints the report and stops the timing, for a command that keeps working
  /// after the compilation. `mojo run` executes the program next: the wall
  /// time of the program does not belong in the report, and an `exit()` from
  /// the program would suppress the report. A command that ends with the
  /// compilation leaves this to the destructor.
  void finish();

  ~MLIRPassTiming() { finish(); }

private:
  TimingReportSink *sink = nullptr;
  mlir::DefaultTimingManager manager;
  // `root` comes after the manager in this declaration. Thus the program
  // destroys `root` first, and the manager sees a scope that is not active if
  // it prints from its own destructor.
  mlir::TimingScope root;
};

/// This class turns on the pass timing of LLVM for one compiler command.
/// `StandardInstrumentations` and the legacy pass manager already install the
/// instrumentation that collects the times. LLVM keeps the times in timer
/// groups that are global to the process. Therefore this class only starts the
/// collection and prints the report.
class LLVMPassTiming {
public:
  // The destructor prints the report and changes state that is global to the
  // process. A copy or a move of this object prints the report again.
  LLVMPassTiming() = default;
  LLVMPassTiming(const LLVMPassTiming &) = delete;
  LLVMPassTiming &operator=(const LLVMPassTiming &) = delete;
  LLVMPassTiming(LLVMPassTiming &&) = delete;
  LLVMPassTiming &operator=(LLVMPassTiming &&) = delete;

  /// Turns on the timing if the command has the `--llvm-timing` option. Also
  /// sets the compilation to one thread, because the timers of LLVM are
  /// global to the process and are not safe for more than one thread. Call
  /// this before the program makes the CPU device from `options`.
  /// `sink` takes the report and must outlive this object.
  void configure(const DriverOptions &driverOptions,
                 KGEN::CompilationOptions &options, TimingReportSink &sink);

  /// Prints the report to stderr and clears the timers. Does nothing if the
  /// timing is off. Does nothing on a second call. The destructor calls this
  /// function for a command that ends with the compilation.
  void finish();

  ~LLVMPassTiming() { finish(); }

private:
  TimingReportSink *sink = nullptr;
  bool enabled = false;
};

/// Wrap a parser invocation to Mojo, populating the necessary parsing context,
/// and attaching post parse metadata. On success, returns the parsed module
/// operation. If `driverOptions` asks to apply fix-its and the parser collects
/// any, they will be applied, and the returned module will be null. If it asks
/// to export them, they will be exported to a YAML file instead. A command that
/// takes `-D` sets `tagEnvironment`, which tags the module with the defines.
ErrorOr<OwningOpRef<ModuleOp>> invokeMojoParser(
    const State &state, const DriverOptions &driverOptions, bool tagEnvironment,
    KGEN::CompilationOptions &compilationOptions,
    KGEN::FrontendOptions frontendOptions, MLIRContext *ctx,
    AsyncRT::CPUDevice &cpuDevice, mlir::TimingScope *timingScope,
    function_ref<OwningOpRef<ModuleOp>(const KGEN::FrontendOptions &,
                                       mlir::TimingScope &)>
        parseFn);

/// Configure cpuDevice options based on compilation options.
/// Currently handles thread pool configuration based on numThreads.
void configureCPUDeviceOptions(AsyncRT::CPUDeviceOptions &cpuDeviceOptions,
                               const KGEN::CompilationOptions &options);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
