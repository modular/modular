//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the main interface for the Mojo Jupyter kernel. It handles
// interacting with the Jupyter kernel protocol and the Mojo LLDB REPL.
//
//===----------------------------------------------------------------------===//

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/SymbolExport.h"
#include "lldb/API/LLDB.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <thread>

#define DEBUG_TYPE "mojo-jupyter"

using namespace lldb;
using namespace lldb_private;
using namespace M;

/// An output function used to send output to the Jupyter kernel. The first
/// argument is the output type, and the second is the output string.
using OutputFn = void (*)(const char *, const char *);

/// TODO(#11553): While the language is private we can't yet define a specific
/// language type. Until then, pretend that we're Go.
static inline constexpr lldb::LanguageType eLanguageTypeMojo =
    lldb::eLanguageTypeGo;

//===----------------------------------------------------------------------===//
// MojoKernel
//===----------------------------------------------------------------------===//

namespace {
/// This class contains all of the various state needed to run the Mojo Jupyter
/// kernel.
class MojoKernel {
public:
  /// This struct represents a single expression evaluation request. It is used
  /// to pass the result of the evaluation back to the caller, which can query
  /// the status of the execution.
  struct ExpressionExecutionState {
    SBError error;
    SBValue result;
    std::thread executionThread;
    std::atomic<bool> finished;
  };

  MojoKernel(OutputFn outputFn) : outputFn(outputFn) {}
  ~MojoKernel() {
    if (process.IsValid())
      process.Kill();
    SBDebugger::Destroy(debugger);
    SBDebugger::Terminate();
  }

  /// Initialize the kernel.
  LogicalResult initialize(const char *mojoReplExe);

  /// Start execution of the given expression string. Returns the state of the
  /// expression execution.
  ExpressionExecutionState *startExecution(const char *expr);

  /// Check if the given expression has finished execution, also taking this
  /// time to flush any collected output.
  bool checkExecutionFinished(ExpressionExecutionState *state);

private:
  /// Initialize the target.
  LogicalResult initializeTarget(const char *mojoReplExe);

  /// Launch the mojo-repl process.
  LogicalResult launchReplProcess();

  /// Report an error to the Jupyter kernel.
  LogicalResult reportKernelError(const Twine &message) {
    sendOutput("error", message.str().c_str());
    return failure();
  }

  /// Send output to the Jupyter kernel.
  void sendOutput(StringRef type, StringRef output) {
    LLVM_DEBUG(llvm::dbgs()
               << "Sending output: " << type << ": " << output << "\n");
    outputFn(type.data(), output.data());
  }

  /// The output function used to send output to the Jupyter kernel.
  OutputFn outputFn;

  /// Various LLDB state used for tracking the repl process.
  SBDebugger debugger;
  SBTarget target;
  SBProcess process;
  SBExpressionOptions exprOpts;
  SBThread mainThread;
};
} // namespace

//===----------------------------------------------------------------------===//
// EntryPoint API
//===----------------------------------------------------------------------===//

MODULAR_EXPORT MojoKernel *initMojoKernel(OutputFn outputFn,
                                          const char *mojoReplExe) {
  std::unique_ptr<MojoKernel> kernel = std::make_unique<MojoKernel>(outputFn);
  if (failed(kernel->initialize(mojoReplExe)))
    return nullptr;
  return kernel.release();
}

MODULAR_EXPORT MojoKernel::ExpressionExecutionState *
startMojoExecution(MojoKernel *kernel, const char *code) {
  return kernel->startExecution(code);
}

MODULAR_EXPORT int
checkMojoExecutionFinished(MojoKernel *kernel,
                           MojoKernel::ExpressionExecutionState *state) {
  return kernel->checkExecutionFinished(state);
}

MODULAR_EXPORT void destroyMojoKernel(MojoKernel *kernel) { delete kernel; }

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

LogicalResult MojoKernel::initialize(const char *mojoReplExe) {
  // Initialize a new debugger instance.
  SBDebugger::Initialize();
  debugger = SBDebugger::Create();
  if (!debugger.IsValid())
    return failure();
  debugger.SetAsync(false);

  // Initialize the Mojo LLDB plugin. We expect the plugin to be adjacent to the
  // MojoJupyter library.
  FileSpec mojoPlugin(Host::GetModuleFileSpecForHostAddress(
      reinterpret_cast<void *>(initMojoKernel)));
  if (!mojoPlugin)
    return reportKernelError("unable to resolve libMojoJupyter location");
  mojoPlugin.SetFilename(
      ("libMojoLLDB" + mojoPlugin.GetFileNameExtension().GetStringRef()).str());
  if (!FileSystem::Instance().Exists(mojoPlugin))
    return reportKernelError("unable to locate libMojoLLDB plugin");
  debugger.HandleCommand(("plugin load " + mojoPlugin.GetPath()).c_str());

  // Initialize the target.
  if (failed(initializeTarget(mojoReplExe)))
    return failure();

  // Create a breakpoint within the repl to act as an anchor for expression
  // evaluation.
  SBBreakpoint mainBreakpoint = target.BreakpointCreateByName(
      "main", target.GetExecutable().GetFilename());
  if (!mainBreakpoint.IsValid())
    return reportKernelError("unable to create breakpoint for repl process");

  // Launch the mojo-repl process.
  if (failed(launchReplProcess()))
    return failure();

  // Initialize the expression options.
  exprOpts = SBExpressionOptions();
  exprOpts.SetLanguage(eLanguageTypeMojo);
  exprOpts.SetUnwindOnError(false);
  exprOpts.SetGenerateDebugInfo(true);

  // Sets an infinite timeout so that users can run arbitrarily long
  // computations.
  exprOpts.SetTimeoutInMicroSeconds(0);
  mainThread = process.GetThreadAtIndex(0);

  LLVM_DEBUG(llvm::dbgs() << "Successfully built Mojo Jupyter kernel\n");
  return success();
}

LogicalResult MojoKernel::initializeTarget(const char *mojoReplExe) {
  // Compute a generic triple for the REPL target.
  llvm::Triple targetTriple = HostInfo::GetArchitecture().GetTriple();
  llvm::SmallString<16> osName;
  llvm::raw_svector_ostream os(osName);

  // Use the most generic sub-architecture.
  targetTriple.setArch(targetTriple.getArch());
  os << llvm::Triple::getOSTypeName(targetTriple.getOS());

  // Override the stub's minimum deployment target to the host os version.
  if (targetTriple.isOSDarwin())
    os << HostInfo::GetOSVersion().getAsString();
  targetTriple.setOSName(os.str());

  // Create a new target for the REPL executable.
  SBError error;
  target = debugger.CreateTarget(
      mojoReplExe, /*target_triple=*/targetTriple.getTriple().c_str(),
      /*platform_name=*/"", /*add_dependent_modules=*/true, error);
  if (!target.IsValid()) {
    return reportKernelError("failed to create target: " +
                             Twine(error.GetCString()));
  }

  return success();
}

LogicalResult MojoKernel::launchReplProcess() {
  auto launchFlags = target.GetLaunchInfo().GetLaunchFlags();
  launchFlags |= lldb::eLaunchFlagDisableASLR;

  // Configure the launch environment to use the target's environment. In
  // addition, we also ensure that the library path includes the directory
  // containing the REPL executable.
  SBEnvironment env = target.GetEnvironment();
  StringRef cwd(target.GetExecutable().GetDirectory());
  env.Set("LD_LIBRARY_PATH", ("$LD_LIBRARY_PATH;" + cwd + "/").str().c_str(),
          /*override=*/true);
  SBStringList envEntries = env.GetEntries();
  std::vector<const char *> envArray;
  for (int i = 0, e = envEntries.GetSize(); i < e; ++i)
    envArray.push_back(envEntries.GetStringAtIndex(i));
  envArray.push_back(nullptr);

  SBListener listener;
  SBError error;
  process = target.Launch(listener, /*argv=*/nullptr, envArray.data(),
                          /*stdin_path=*/nullptr, /*stdout_path=*/nullptr,
                          /*stderr_path=*/nullptr, cwd.data(), launchFlags,
                          /*stop_at_entry=*/false, error);
  if (!process.IsValid() || process.GetState() != eStateStopped) {
    return reportKernelError("Failed to launch `mojo-repl` process: " +
                             Twine(error.GetCString()));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

MojoKernel::ExpressionExecutionState *
MojoKernel::startExecution(const char *expr) {
  ExpressionExecutionState *state = new ExpressionExecutionState();

  // Start execution of the expression in a separate thread, so that way the
  // calling client can control waiting for the expression to complete.
  state->executionThread =
      std::thread([this, state, expr = std::string(expr)]() mutable {
        LLVM_DEBUG(llvm::dbgs() << "Executing expression: " << expr << "\n");
        state->result = target.EvaluateExpression(expr.data(), exprOpts);
        state->error = state->result.GetError();
        state->finished = true;
      });

  return state;
}

bool MojoKernel::checkExecutionFinished(ExpressionExecutionState *state) {
  // Flush out any pending output.
  char outputBuffer[1024];

  // Read stdout from the process.
  while (int readLen = process.GetSTDOUT(outputBuffer, 1023)) {
    outputBuffer[readLen] = '\0';
    LLVM_DEBUG(llvm::dbgs()
               << "stdout: " << readLen << " : " << outputBuffer << "\n");
    sendOutput("stdout", outputBuffer);
  }
  // Read stderr from the process.
  while (int readLen = process.GetSTDERR(outputBuffer, 1023)) {
    outputBuffer[readLen] = '\0';
    LLVM_DEBUG(llvm::dbgs()
               << "stderr: " << readLen << " : " << outputBuffer << "\n");
    sendOutput("stderr", outputBuffer);
  }

  // Check to see if the expression is still executing.
  if (!state->finished)
    return false;

  // The expression has finished executing, process the results.
  LLVM_DEBUG(llvm::dbgs() << "Finished executing expression\n");

  // Process the result.
  auto errorType = state->error.GetType();
  if (errorType == eErrorTypeInvalid)
    sendOutput("stdout", state->result.GetObjectDescription());
  else if (errorType != eErrorTypeGeneric)
    sendOutput("stderr", state->error.GetCString());
  else
    state->error.Clear();

  // Clean up the state now that we're done with it.
  state->executionThread.join();
  delete state;
  return true;
}
