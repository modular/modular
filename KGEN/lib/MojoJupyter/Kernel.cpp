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

#include "../MojoLLDB/ExpressionParser/MojoExpressionVariable.h"
#include "../MojoLLDB/TypeSystem/MojoTypeSystem.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/SymbolExport.h"
#include "lldb/API/LLDB.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Listener.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <thread>

#define DEBUG_TYPE "mojo-jupyter"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN::Mojo;

/// An output function used to send output to the Jupyter kernel. The first
/// argument is the output type, and the second is the output string.
using OutputFn = void (*)(const char *, const char *);

namespace {
//===----------------------------------------------------------------------===//
// MojoTarget
//===----------------------------------------------------------------------===//

struct MojoTarget : public SBTarget {
  using SBTarget::SBTarget;
  using SBTarget::operator=;

  /// Return the persistent expression state for Mojo.
  MojoPersistentExpressionState *GetPersistentExpressionState() {
    return static_cast<MojoPersistentExpressionState *>(
        GetSP()->GetPersistentExpressionStateForLanguage(eLanguageTypeMojo));
  }

  MojoTypeSystem &getMojoTypeSystem() {
    if (auto typeSystemOr =
            GetSP()->GetScratchTypeSystemForLanguage(eLanguageTypeMojo))
      return *static_cast<MojoTypeSystem *>(typeSystemOr.get().get());
    llvm::report_fatal_error(
        "The Mojo type system plug-in must have already been registered.");
  }
};

//===----------------------------------------------------------------------===//
// MojoExpressionEvaluationOptions
//===----------------------------------------------------------------------===//

struct MojoExpressionEvaluationOptions : public SBExpressionOptions {
  MojoExpressionEvaluationOptions() {
    SetLanguage(eLanguageTypeMojo);
    SetUnwindOnError(false);
    SetGenerateDebugInfo(true);

    // Sets an infinite timeout so that users can run arbitrarily long
    // computations.
    SetTimeoutInMicroSeconds(0);

    // TODO: This should be part of the public API, but for now we need to set
    // it via the private API.
    ref().SetREPLEnabled(true);
  }
};

//===----------------------------------------------------------------------===//
// MojoKernel
//===----------------------------------------------------------------------===//

/// This class contains all of the various state needed to run the Mojo Jupyter
/// kernel.
class MojoKernel {
public:
  /// Information related to a specific kernel cell.
  struct KernelCellState {
    KernelCellState(StringRef id) : id(id) {}

    /// The string identifier of the cell.
    StringRef id;

    /// The index of the expression instance associated with this cell within
    /// the Mojo persistent state, or nullopt if the cell was not successfully
    /// executed.
    std::optional<unsigned> replExprIdx;
  };

  /// This struct represents a single expression evaluation request. It is used
  /// to pass the result of the evaluation back to the caller, which can query
  /// the status of the execution.
  struct ExpressionExecutionState {
    SBError error;
    SBValue result;
    std::thread executionThread;
    std::atomic<bool> finished;
    KernelCellState *cellState = nullptr;
  };

  MojoKernel(OutputFn outputFn)
      : outputFn(outputFn), mojoTypeSystemListener(Listener::MakeListener(
                                "mojo-type-system.listener")) {}
  ~MojoKernel() {
    if (process.IsValid())
      process.Kill();
    SBDebugger::Destroy(debugger);
    SBDebugger::Terminate();
  }

  /// Initialize the kernel.
  LogicalResult initialize(const char *mojoReplExe);

  /// Start execution of the given cell identifier and expression string.
  /// Returns the state of the expression execution.
  ExpressionExecutionState *startExecution(const char *cellId,
                                           const char *expr);

  /// Check if the given expression has finished execution, also taking this
  /// time to flush any collected output.
  bool checkExecutionFinished(ExpressionExecutionState *state);

private:
  /// Initialize the target.
  LogicalResult initializeTarget(const char *mojoReplExe);

  /// Launch the mojo-repl-entry-point process.
  LogicalResult launchReplProcess();

  /// Report an error to the Jupyter kernel.
  LogicalResult reportKernelError(const Twine &message) {
    llvm::errs() << "error: " << message << "\n";
    sendOutput("error", message.str().c_str());
    return failure();
  }

  /// Send output to the Jupyter kernel.
  void sendOutput(StringRef type, StringRef output) {
    LLVM_DEBUG(llvm::dbgs()
               << "Sending output: " << type << ": " << output << "\n");
    outputFn(type.data(), output.data());
  }

  /// Flush the LLDB output streams associated within the given execution state.
  void flushLLDBStreams(ExpressionExecutionState *state);

  /// Initialize the given cell for execution. Returns the associated cell
  /// state, or nullptr if the cell was invalid.
  KernelCellState *initializeCellForExecution(const char *cellId);

  /// The output function used to send output to the Jupyter kernel.
  OutputFn outputFn;

  /// Various LLDB state used for tracking the repl process.
  SBDebugger debugger;
  MojoTarget target;
  SBProcess process;
  MojoExpressionEvaluationOptions exprOpts;
  SBThread mainThread;
  ListenerSP mojoTypeSystemListener;

  /// The mojo persistent expression state.
  MojoPersistentExpressionState *exprState = nullptr;

  /// An ordered list containing information about each of the cells that have
  /// been executed.
  std::vector<std::unique_ptr<KernelCellState>> cells;
  llvm::StringMap<unsigned> cellIdToIndex;
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
startMojoExecution(MojoKernel *kernel, const char *cellId, const char *code) {
  return kernel->startExecution(cellId, code);
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

  // This will log to LLDB's stderr, which the notebook server will pick up (but
  // not the notebook).
  debugger.HandleCommand("log enable lldb expr");

  // Initialize the target.
  if (failed(initializeTarget(mojoReplExe)))
    return failure();

  // Create a breakpoint within the repl to act as an anchor for expression
  // evaluation.
  SBBreakpoint mainBreakpoint = target.BreakpointCreateByName(
      "main", target.GetExecutable().GetFilename());
  if (!mainBreakpoint.IsValid())
    return reportKernelError("unable to create breakpoint for repl process");

  // Launch the mojo-repl-entry-point process.
  if (failed(launchReplProcess()))
    return failure();

  // Sets an infinite timeout so that users can run arbitrarily long
  // computations.
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
  exprState = target.GetPersistentExpressionState();

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
    return reportKernelError(
        "Failed to launch `mojo-repl-entry-point` process: " +
        Twine(error.GetCString()));
  }

  target.getMojoTypeSystem().AddListener(
      mojoTypeSystemListener,
      KGEN::Mojo::MojoTypeSystem::eBroadcastUserMessage);

  return success();
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

MojoKernel::ExpressionExecutionState *
MojoKernel::startExecution(const char *cellId, const char *expr) {
  ExpressionExecutionState *state = new ExpressionExecutionState();
  state->cellState = initializeCellForExecution(cellId);

  // Start execution of the expression in a separate thread, so that way the
  // calling client can control waiting for the expression to complete.
  state->executionThread =
      std::thread([this, state, expr = std::string(expr)]() mutable {
        LLVM_DEBUG(llvm::dbgs() << "Executing expression: " << expr << "\n");
        unsigned exprInstIdx = exprState->getNumExpressionInstances();
        state->result = target.EvaluateExpression(expr.data(), exprOpts);
        state->error = state->result.GetError();
        state->finished = true;

        // If the REPL pushed a new expression state, associate it with the
        // cell.
        unsigned newExprInstIdx = exprState->getNumExpressionInstances();
        if (state->cellState && newExprInstIdx != exprInstIdx)
          state->cellState->replExprIdx = exprInstIdx;
      });

  return state;
}

void MojoKernel::flushLLDBStreams(ExpressionExecutionState *state) {
  // Reading the following streams from LLDB is thread safe becaause each reader
  // has its own mutex.

  // Flush type system messages.
  lldb::EventSP event;

  // The following gets the stream of events without timeout. All the messages
  // will be read eventually anyway.
  while (mojoTypeSystemListener->GetEvent(event, std::chrono::seconds(0))) {
    size_t readLen = EventDataBytes::GetByteSizeFromEvent(event.get());
    const char *rawData = static_cast<const char *>(
        EventDataBytes::GetBytesFromEvent(event.get()));
    StringRef data(rawData, readLen);
    LLVM_DEBUG(llvm::dbgs()
               << "type system message: " << readLen << " : " << data << "\n");
    // We need to ensure that the output is null terminated.
    sendOutput("stderr", data.str());
  }

  char outputBuffer[1024];

  // Read stdout from the process.
  while (int readLen = process.GetSTDOUT(outputBuffer, 1023)) {
    outputBuffer[readLen] = '\0';
    StringRef data(outputBuffer, readLen);
    LLVM_DEBUG(llvm::dbgs() << "stdout: " << readLen << " : " << data << "\n");
    sendOutput("stdout", data);
  }
  // Read stderr from the process.
  while (int readLen = process.GetSTDERR(outputBuffer, 1024)) {
    outputBuffer[readLen] = '\0';
    StringRef data(outputBuffer, readLen);
    LLVM_DEBUG(llvm::dbgs() << "stderr: " << readLen << " : " << data << "\n");
    sendOutput("stderr", data);
  }
}

bool MojoKernel::checkExecutionFinished(ExpressionExecutionState *state) {
  // Check to see if the expression is still executing.
  if (!state->finished) {
    flushLLDBStreams(state);
    return false;
  }
  flushLLDBStreams(state);

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

MojoKernel::KernelCellState *
MojoKernel::initializeCellForExecution(const char *cellId) {
  if (!cellId)
    return nullptr;
  auto [cellIt, inserted] = cellIdToIndex.insert({cellId, cells.size()});

  // If this is a new cell, we just need to construct a new state.
  if (inserted) {
    return &*cells.emplace_back(
        std::make_unique<KernelCellState>(cellIt->first()));
  }
  KernelCellState *cellState = cells[cellIt->second].get();
  unsigned nextCellIndex = cellIt->second + 1;

  // Otherwise, this is a pre-existing cell. Reset the REPL state to just before
  // this cell.
  bool shouldResetExprState = true;
  auto resetExprState = [&](std::optional<unsigned> exprIdx) {
    if (exprIdx && std::exchange(shouldResetExprState, false))
      exprState->resetStateToBeforeExpressionInstance(*exprIdx);
  };

  // Reset any REPL state associated with cells starting with this one,
  // completely dropping follow-on cells.
  resetExprState(cellState->replExprIdx);
  for (auto &cellState : llvm::drop_begin(cells, nextCellIndex)) {
    resetExprState(cellState->replExprIdx);
    cellIdToIndex.erase(cellState->id);
  }
  cells.resize(nextCellIndex);
  return cellState;
}
