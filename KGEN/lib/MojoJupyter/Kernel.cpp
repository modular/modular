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
#include "../MojoLLDB/REPL/MojoREPL.h"
#include "../MojoLLDB/ScriptingBridge/SBClassUtils.h"
#include "../MojoLLDB/TypeSystem/MojoTypeSystem.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/STLExtras.h"
#include "Support/SymbolExport.h"
#include "lldb/API/LLDB.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Listener.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <filesystem>
#include <thread>

#define DEBUG_TYPE "mojo-jupyter"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN::Mojo;

/// An output function used to send output to the Jupyter kernel. The first
/// argument is the output type, and the second is the output string.
using OutputFn = void (*)(const char *, const char *);

/// Return the persistent expression state for Mojo.
static MojoPersistentExpressionState *
getPersistentExpressionState(const TargetSP &target) {
  return static_cast<MojoPersistentExpressionState *>(
      target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeMojo));
}

static MojoTypeSystem &getMojoTypeSystem(const TargetSP &target) {
  if (auto typeSystemOr =
          target->GetScratchTypeSystemForLanguage(lldb::eLanguageTypeMojo))
    return *static_cast<MojoTypeSystem *>(typeSystemOr.get().get());
  llvm::report_fatal_error(
      "The Mojo type system plug-in must have already been registered.");
}

/// These values were chosen because both `finished` and `error` are 'truthy',
/// and indicate that the execution has finished. Therefore, only clients that
/// care if there was an error have to do anything with it.
///
/// When updating this enum, please take care to also update the enums in
/// mojo-jupyter-executor/main.cpp and mojokernel.py.
enum ExecutionFinishedState : int {
  kNotFinished = 0,
  kFinishedSuccessfully = 1,
  kFinishedError = 2,
};

//===----------------------------------------------------------------------===//
// MojoExpressionEvaluationOptions
//===----------------------------------------------------------------------===//

namespace {
struct MojoExpressionEvaluationOptions : public SBExpressionOptions {
  MojoExpressionEvaluationOptions() {
    SetLanguage(lldb::eLanguageTypeMojo);
    SetUnwindOnError(false);
    SetGenerateDebugInfo(true);

    // Sets an infinite timeout so that users can run arbitrarily long
    // computations.
    SetTimeoutInMicroSeconds(0);

    ref().SetREPLEnabled(true);
    ref().SetColorizeErrors(true);
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

    KernelCellState(const KernelCellState &) = delete;
    KernelCellState &operator=(const KernelCellState &) = delete;

    /// The string identifier of the cell.
    std::string id;

    /// This is a list of debug messages that we'll only flush to the kernel's
    /// stderr if we are told to do so.
    std::deque<std::pair<MojoTypeSystem::MessageKind, std::string>>
        debugMessages;
  };

  /// This struct represents a single expression evaluation request. It is used
  /// to pass the result of the evaluation back to the caller, which can query
  /// the status of the execution.
  struct ExpressionExecutionState {
    ExpressionExecutionState(KernelCellState &cellState)
        : finished(false), cellState(cellState) {}
    SBError error;
    SBValue result;
    std::thread executionThread;
    std::atomic<bool> finished;
    KernelCellState &cellState;
  };

  MojoKernel(OutputFn outputFn)
      : outputFn(outputFn), mojoTypeSystemListener(Listener::MakeListener(
                                "mojo-type-system.listener")) {
    // Check for an environment variable we'll use to specify the log file path.
    std::optional<std::string> logFilePath =
        llvm::sys::Process::GetEnv("MOJO_JUPYTER_LOG_FILE");
    if (!logFilePath.has_value()) {
      // If we don't have a log file path, simply log to stderr.
      logStream =
          ConditionallyOwnedPointer<llvm::raw_ostream>::borrow(&llvm::errs());
      return;
    }

    // We have a path, log to a file.
    std::error_code ec;
    logStream = ConditionallyOwnedPointer<llvm::raw_ostream>::allocate<
        llvm::raw_fd_ostream>(*logFilePath, ec);
    logStream->SetUnbuffered();

    // We must not error opening the log file provided.
    if (ec)
      llvm::report_fatal_error("Error opening " + Twine(*logFilePath) +
                               " for logging: " + ec.message());
  }

  ~MojoKernel() {
    if (process->IsValid())
      process->Destroy(/*force_kill=*/true);
    SBDebugger sbdebugger = SBDebuggerUtils::create(debugger);
    SBDebugger::Destroy(sbdebugger);
    SBDebugger::Terminate();
  }

  /// Initialize the kernel.
  ///
  /// If `lldbInitFile` is not null, LLDB will silently execute all the commands
  /// in this file upon initialization of the kernel.
  LogicalResult initialize(const char *mojoReplExe, const char *lldbInitFile);

  /// Start execution of the given cell identifier and expression string.
  /// `storeHistory` indicates if variables and state from this expression
  /// should be persisted. Returns the state of the expression execution.
  void startExecution(StringRef cellId, const char *expr, int storeHistory);

  /// Check if the current expression has finished execution, also taking this
  /// time to flush any collected output.
  ExecutionFinishedState checkExecutionFinished();

  /// Interrupt the currently running execution.
  void interruptExecution();

private:
  /// Initialize the target.
  LogicalResult initializeTarget(const char *mojoReplExe);

  /// Launch the mojo-repl-entry-point process.
  LogicalResult launchReplProcess();

  /// Report an error to the Jupyter kernel.
  LogicalResult reportKernelError(const Twine &message) {
    llvm::errs() << "error: " << message << "\n";
    sendOutput("error", message.str());
    return failure();
  }

  /// Send output to the Jupyter kernel.
  void sendOutput(StringRef type, StringRef output) {
    LLVM_DEBUG(llvm::dbgs()
               << "Sending output: " << type << ": " << output << "\n");
    outputFn(type.data(), output.data());
  }

  /// Flush the LLDB output streams associated within the current execution
  /// state.
  void flushLLDBStreams();

  /// Initialize the given cell for execution. Returns the associated cell
  /// state, or nullptr if the cell was invalid.
  KernelCellState &initializeCellForExecution(StringRef cellId) {
    auto [it, inserted] = cells.insert({cellId, nullptr});
    if (inserted)
      it->second = std::make_unique<KernelCellState>(it->first());
    return *it->second;
  }

  /// The output function used to send output to the Jupyter kernel.
  OutputFn outputFn;

  /// Various LLDB state used for tracking the repl process.
  DebuggerSP debugger;
  TargetSP target;
  ProcessSP process;
  MojoExpressionEvaluationOptions exprOpts;
  ThreadSP mainThread;
  ListenerSP mojoTypeSystemListener;

  /// The stream to write logs to. This may point to a file if the
  /// `MOJO_JUPYTER_LOG_FILE` env variable is specified, otherwise it points to
  /// stderr.
  ConditionallyOwnedPointer<llvm::raw_ostream> logStream;

  /// The mojo persistent expression state.
  MojoPersistentExpressionState *exprState = nullptr;

  /// The current execution state, or nullopt if no execution is currently
  /// happening.
  std::optional<ExpressionExecutionState> executionState;

  /// Information about each of the cells that have been executed.
  llvm::StringMap<std::unique_ptr<KernelCellState>> cells;
};
} // namespace

//===----------------------------------------------------------------------===//
// EntryPoint API
//===----------------------------------------------------------------------===//

MODULAR_EXPORT MojoKernel *initMojoKernel(OutputFn outputFn,
                                          const char *mojoReplExe,
                                          const char *lldbInitFile) {
  std::unique_ptr<MojoKernel> kernel = std::make_unique<MojoKernel>(outputFn);
  if (failed(kernel->initialize(mojoReplExe, lldbInitFile)))
    return nullptr;
  return kernel.release();
}

MODULAR_EXPORT void startMojoExecution(MojoKernel *kernel, const char *cellId,
                                       const char *code, int storeHistory) {
  kernel->startExecution(cellId, code, storeHistory);
}

MODULAR_EXPORT int checkMojoExecutionFinished(MojoKernel *kernel) {
  return kernel->checkExecutionFinished();
}

MODULAR_EXPORT void interruptMojoExecution(MojoKernel *kernel) {
  kernel->interruptExecution();
}

MODULAR_EXPORT void destroyMojoKernel(MojoKernel *kernel) { delete kernel; }

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// We want to restrict the set of LLDB commands that can be executed on the
/// notebook as a way to prevent external users from affecting the host
/// environment (e.g. killing or spawning processes, inspecting the entry-point,
/// etc.)
static void removeUnwantedCommands(Debugger &debugger) {
  auto isAllowed = [](const CommandObjectSP &obj) {
    /// The following list can grow as needed, but just be mindful of any
    /// possible vulnerability, as LLDB has more permissions than regular
    /// processes.
    static auto kAllowedCommands = {
        "apropos",
        "help",
        "mojo",
        // This is very useful for us to debug issues in the expression
        // evaluator.
        "log",
    };
    return llvm::any_of(kAllowedCommands, [&](StringRef allowed) {
      return obj->GetCommandName().starts_with(allowed);
    });
  };

  CommandInterpreter &interpreter = debugger.GetCommandInterpreter();

  CommandObject::CommandMap aliases = interpreter.GetAliases();
  for (auto &[name, obj] : aliases) {
    CommandObjectSP actualCommand =
        static_cast<CommandAlias *>(obj.get())->GetUnderlyingCommand();
    if (!isAllowed(actualCommand))
      interpreter.RemoveAlias(name);
  }

  CommandObject::CommandMap commands = interpreter.GetCommands();
  for (auto &[name, obj] : commands) {
    if (!isAllowed(obj))
      interpreter.RemoveCommand(name);
  }
}

static void runLLDBInitFile(Debugger &debugger, FileSpec lldbInitFile) {
  CommandInterpreterRunOptions options;
  CommandReturnObject result(/*colors=*/false);
  options.SetAddToHistory(false);
  options.SetEchoCommands(false);
  options.SetPrintErrors(true);
  options.SetSilent(true);
  debugger.GetCommandInterpreter().HandleCommandsFromFile(lldbInitFile, options,
                                                          result);
  if (!result.Succeeded()) {
    llvm::errs() << result.GetOutputData() << "\n";
    llvm::errs() << result.GetErrorData() << "\n";
    exit(EXIT_FAILURE);
  }
}

LogicalResult MojoKernel::initialize(const char *mojoReplExe,
                                     const char *lldbInitFile) {
  // Initialize a new debugger instance.
  // We need to initialize with SBDebugger because that's the only way we can
  // support loading public plugins like MojoLLDB.
  SBDebugger::Initialize();
  debugger = Debugger::CreateInstance();
  debugger->SetAsyncExecution(false);

  // If we got an LLDB init file, we execute it before anything else.
  if (lldbInitFile)
    runLLDBInitFile(*debugger, FileSpec(lldbInitFile));

  // For security reasons on public Jupyter notebooks, we want to remove some
  // commands that might give users ways to perform unwanted actions on the
  // host.
  removeUnwantedCommands(*debugger);

  // Initialize the Mojo LLDB plugin. We expect the plugin to be adjacent to the
  // MojoJupyter library.
  FileSpec mojoPlugin(Host::GetModuleFileSpecForHostAddress(
      reinterpret_cast<void *>(initMojoKernel)));
  if (!mojoPlugin)
    return reportKernelError("unable to resolve libMojoJupyter location");

  StringRef libExt = mojoPlugin.GetFilename().GetStringRef().split('.').second;
  mojoPlugin.SetFilename(("libMojoLLDB." + libExt).str());
  if (!FileSystem::Instance().Exists(mojoPlugin))
    return reportKernelError("unable to locate libMojoLLDB plugin");

  CommandReturnObject result(/*colors=*/false);
  Status err;
  if (!debugger->LoadPlugin(mojoPlugin, err))
    return reportKernelError(err.AsCString());

  // Initialize the target.
  if (failed(initializeTarget(mojoReplExe)))
    return failure();

  // Create a breakpoint within the repl to act as an anchor for expression
  // evaluation.
  BreakpointSP mainBreakpoint = target->CreateBreakpoint(
      /*all modules*/ nullptr, /*all sources*/ nullptr, "main",
      eFunctionNameTypeAuto, eLanguageTypeUnknown,
      /*offset=*/0,
      /*skip_prologue=*/eLazyBoolCalculate,
      /*internal=*/false, /*hardware=*/false);
  if (!mainBreakpoint)
    return reportKernelError("unable to create breakpoint for repl process");

  // Launch the mojo-repl-entry-point process.
  if (failed(launchReplProcess()))
    return failure();
  process = target->GetProcessSP();

  // Sets an infinite timeout so that users can run arbitrarily long
  // computations.
  mainThread = process->GetThreadList().GetThreadAtIndex(0);

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
  debugger->GetTargetList().CreateTarget(
      *debugger, mojoReplExe,
      /*target_triple=*/targetTriple.getTriple().c_str(),
      /*add_dependent_modules=*/eLoadDependentsYes,
      /*platform_options=*/nullptr, target);

  if (!target)
    return reportKernelError("failed to create target: invalid debugger");
  exprState = getPersistentExpressionState(target);

  return success();
}

LogicalResult MojoKernel::launchReplProcess() {
  if (llvm::Error err = MojoREPL::launchEntryPointProcess(*target, *debugger)) {
    return reportKernelError(
        "Failed to launch `mojo-repl-entry-point` process: " +
        llvm::toString(std::move(err)));
  }
  getMojoTypeSystem(target).AddListener(mojoTypeSystemListener,
                                        MojoTypeSystem::eAllMessagesMask);

  return success();
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

void MojoKernel::startExecution(StringRef cellId, const char *expr,
                                int storeHistory) {
  executionState.emplace(initializeCellForExecution(cellId));

  // Start execution of the expression in a separate thread, so that way the
  // calling client can control waiting for the expression to complete.
  executionState->executionThread =
      std::thread([this, storeHistory, expr = std::string(expr)]() mutable {
        LLVM_DEBUG(llvm::dbgs() << "Executing expression: " << expr << "\n");

        // If the expression starts with `:`, then it is an LLDB command,
        // otherwise it is a Mojo expression.
        SBValue value;
        if (StringRef command(expr); command.consume_front(":")) {
          CommandReturnObject result(/*colors=*/false);
          target->GetDebugger().GetCommandInterpreter().HandleCommand(
              command.rtrim().str().c_str(),
              /*add_to_history=*/lldb_private::eLazyBoolNo, result);
          sendOutput("output", result.GetOutputData());
          if (!result.GetErrorData().empty())
            sendOutput("error", result.GetErrorData());
        } else {
          MojoExpressionEvaluationOptions options = exprOpts;
          if (!storeHistory)
            options.SetSuppressPersistentResult(true);

          value = SBTargetUtils::create(target).EvaluateExpression(expr.data(),
                                                                   options);
        }

        executionState->result = value;
        executionState->error = value.GetError();

        // Mark the execution as finished.
        executionState->finished = true;
      });
}

void MojoKernel::flushLLDBStreams() {
  // Reading the following streams from LLDB is thread safe becaause each reader
  // has its own mutex.

  // Flush type system messages.
  lldb::EventSP event;

  // Various logging utilities (like CloudWatch) parse JSON automatically so we
  // should use that for structured logging.
  auto reportMessage = [&](StringRef type, StringRef message) {
    if (llvm::sys::Process::GetEnv("MOJO_JUPYTER_JSON_LOGS")) {
      llvm::json::OStream j(*logStream);
      // Produce `{"type": <type>, "message": <message>}`
      j.object([&]() {
        j.attribute("type", type);
        j.attribute("message", message);
      });
      *logStream << "\n";
    } else {
      *logStream << '[' << type << "] " << message << "\n";
    }
  };

  // The following gets the stream of events without timeout. All the messages
  // will be read eventually anyway.
  while (mojoTypeSystemListener->GetEvent(event, std::chrono::seconds(0))) {
    MojoTypeSystem::handleEvent(
        event, executionState->cellState.debugMessages, reportMessage,
        [&](StringRef msg) { sendOutput("stderr", msg); });
    event->Clear();
  }

  char outputBuffer[1024];

  // Read stdout from the process.
  Status unused;
  while (int readLen = process->GetSTDOUT(outputBuffer, 1023, unused)) {
    outputBuffer[readLen] = '\0';
    StringRef data(outputBuffer, readLen);
    LLVM_DEBUG(llvm::dbgs() << "stdout: " << readLen << " : " << data << "\n");
    sendOutput("stdout", data);
  }
  // Read stderr from the process.
  while (int readLen = process->GetSTDERR(outputBuffer, 1024, unused)) {
    outputBuffer[readLen] = '\0';
    StringRef data(outputBuffer, readLen);
    LLVM_DEBUG(llvm::dbgs() << "stderr: " << readLen << " : " << data << "\n");
    sendOutput("stderr", data);
  }
}

ExecutionFinishedState MojoKernel::checkExecutionFinished() {
  if (!executionState)
    return kFinishedSuccessfully;

  // Check to see if the expression is still executing.
  if (!executionState->finished) {
    flushLLDBStreams();
    return kNotFinished;
  }
  flushLLDBStreams();

  // The expression has finished executing, process the results.
  LLVM_DEBUG(llvm::dbgs() << "Finished executing expression\n");

  ExecutionFinishedState finishState = kFinishedSuccessfully;
  // Process the result.
  auto errorType = executionState->error.GetType();
  if (errorType == eErrorTypeInvalid) {
    sendOutput("stdout", executionState->result.GetObjectDescription());
  } else if (errorType != eErrorTypeGeneric) {
    StringRef executionError(executionState->error.GetCString());

    // If the expression failed to parse, LLDB adds in an extra message, strip
    // that out.
    executionError.consume_front("expression failed to parse:\n");

    // If the output is simply "unknown error", this indicates that LLDB didn't
    // have a diagnostic for the specific problem. In these cases, the REPL
    // ensures that the user is alerted to the problem, so there isn't a need to
    // add the unhelpful error message.
    if (executionError != "unknown error")
      sendOutput("stderr", executionError);

    // Set the finish state to an error.
    finishState = kFinishedError;
  } else {
    // eErrorTypeGeneric can be thrown if the expression doesn't have a result,
    // even if it succeeded.
    // TODO: Revisit this once we have more TypeSystem functionality
    //   implemented.
    executionState->error.Clear();
  }

  // Clean up the state now that we're done with it.
  executionState->executionThread.join();
  executionState.reset();
  return finishState;
}

void MojoKernel::interruptExecution() { process->SendAsyncInterrupt(); }
