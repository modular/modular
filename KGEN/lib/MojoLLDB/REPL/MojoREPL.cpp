//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoREPL.h"
#include "../Plugin.h"
#include "../TypeSystem/MojoTypeSystem.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBProcess.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

#if defined(_WIN32)
#define REPL_ENTRY_POINT_BIN "mojo-repl-entry-point.exe"
#else
#define REPL_ENTRY_POINT_BIN "mojo-repl-entry-point"
#endif

static llvm::Error createStringError(StringRef message) {
  return llvm::make_error<llvm::StringError>(message,
                                             llvm::inconvertibleErrorCode());
}
template <typename... Args>
static llvm::Error createStringError(const char *format, Args &&...args) {
  return createStringError(
      llvm::formatv(format, std::forward<Args>(args)...).str());
}

//===----------------------------------------------------------------------===//
// Target event listening
//===----------------------------------------------------------------------===//

static bool shouldStopListeningToEvents(lldb::StateType state) {
  switch (state) {
  case lldb::eStateConnected:
  case lldb::eStateAttaching:
  case lldb::eStateLaunching:
  case lldb::eStateStepping:
  case lldb::eStateSuspended:
  case lldb::eStateStopped:
  case lldb::eStateRunning:
  case lldb::eStateCrashed:
    return false;
  case lldb::eStateInvalid:
  case lldb::eStateDetached:
  case lldb::eStateExited:
  case lldb::eStateUnloaded:
    // Only in these states we can't execute more expressions.
    return true;
  }
}

static void flushInferiorStderrAndStdout(lldb::ProcessSP &process) {
  constexpr size_t kBufferSize = 1024;
  char buffer[kBufferSize];
  size_t count;
  lldb_private::Debugger &debugger = process->GetTarget().GetDebugger();
  Status status;
  {
    lldb::StreamSP out(debugger.GetAsyncOutputStream());
    while ((count = process->GetSTDOUT(buffer, kBufferSize - 1, status)) > 0) {
      buffer[count] = '\0';
      out->Write(buffer, count);
    }
    out->Flush();
  }
  {
    lldb::StreamSP err(debugger.GetAsyncErrorStream());
    while ((count = process->GetSTDERR(buffer, kBufferSize - 1, status)) > 0) {
      buffer[count] = '\0';
      err->Write(buffer, count);
    }
    err->Flush();
  }
}

static void eventThreadFunction(const lldb::TargetSP &target,
                                const std::atomic_bool &stopEventThread) {
  lldb::ProcessSP process(target->GetProcessSP());
  assert(process->IsValid() &&
         "A valid process should already exist for the REPL");
  // Get a pointer to the mojo type system. We need that to read the various log
  // messages.
  auto typeSystemOr =
      target->GetScratchTypeSystemForLanguage(eLanguageTypeMojo);
  if (!typeSystemOr)
    llvm::report_fatal_error("must be able to get the mojo type system");

  std::shared_ptr<MojoTypeSystem> typeSystem =
      std::static_pointer_cast<MojoTypeSystem>(*typeSystemOr);
  if (!typeSystem)
    llvm::report_fatal_error("must be able to get the mojo type system");

  lldb::ListenerSP processListener =
      Listener::MakeListener("mojo-repl.process-listener");
  process->AddListener(processListener, Process::eBroadcastBitStateChanged |
                                            Process::eBroadcastBitSTDOUT |
                                            Process::eBroadcastBitSTDERR);
  lldb::ListenerSP typeSystemListener =
      Listener::MakeListener("mojo-repl.type-system-listener");
  typeSystem->AddListener(typeSystemListener, MojoTypeSystem::eAllMessagesMask);

  // Construct the debug message cache, and pull out the error raw_ostream now.
  std::deque<std::pair<MojoTypeSystem::MessageKind, std::string>> debugMessages;
  auto errorStream = process->GetTarget().GetDebugger().GetAsyncErrorStream();
  // Report a message to the error stream.
  auto reportMessage = [errorStream](StringRef type, StringRef message) {
    // If the LLDB Expression logs are enabled, we should send our message
    // there. This has the benefit of being able to automatically send our logs
    // to a file if the LLDB log has been configured to do so. And if not, they
    // will appear in the error stream anyway.
    if (Log *log = GetLog(LLDBLog::Expressions))
      LLDB_LOG(log, "[{0}] {1}", type, message);
    else
      errorStream->AsRawOstream() << "[" << type << "] " << message << "\n";
  };
  auto sendUserOutput = [errorStream](StringRef message) {
    errorStream->AsRawOstream() << "[User] " << message << "\n";
  };

  // Single run that processes pending events. It returns true if the listener
  // loop should resume, or false if the inferior can't execute more
  // expressions and the loop should stop.
  auto processEvents = [&]() -> bool {
    lldb::EventSP event;
    while (typeSystemListener->GetEvent(event, std::chrono::seconds(0))) {
      // Handle the mojo type system events by logging them to the debugger
      // stderr.
      MojoTypeSystem::handleEvent(event, debugMessages, reportMessage,
                                  sendUserOutput);

      // Flush the error stream immediately - otherwise it only shows up when
      // we shut down the repl.
      errorStream->Flush();
    }
    while (processListener->GetEvent(event, std::chrono::seconds(0))) {
      const uint32_t eventMask = event->GetType();
      if (eventMask & Process::eBroadcastBitStateChanged) {
        if (shouldStopListeningToEvents(
                Process::ProcessEventData::GetStateFromEvent(event.get()))) {
          return false;
        }
      } else if ((eventMask & Process::eBroadcastBitSTDOUT) ||
                 (eventMask & Process::eBroadcastBitSTDERR)) {
        flushInferiorStderrAndStdout(process);
      }
    }
    return true;
  };

  while (!stopEventThread && processEvents())
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // We fetch the events one last time in case the processes emitted some after
  // the previous loop was told to stop by the REPL's destructor.
  processEvents();
}

//===----------------------------------------------------------------------===//
// MojoREPL
//===----------------------------------------------------------------------===//

MojoREPL::MojoREPL(Target &target) : REPL(eKindGo, target) {
  eventThread = std::thread([this] {
    eventThreadFunction(m_target.shared_from_this(), stopEventThread);
  });
}

MojoREPL::~MojoREPL() {
  if (eventThread.joinable()) {
    stopEventThread = true;
    eventThread.join();
  }
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Create a repl instance for a given target. `replOptions` contains a set of
/// options to be passed to the repl.
static llvm::Expected<lldb::REPLSP>
createInstanceFromTarget(Target &target, const char *replOptions) {
  // Sanity check the target to make sure a REPL would work here.
  if (!target.GetProcessSP() || !target.GetProcessSP()->IsAlive()) {
    return createStringError(
        "can't launch a Mojo REPL without a running process");
  }

  lldb::REPLSP repl = std::make_shared<MojoREPL>(target);
  repl->SetCompilerOptions(replOptions);
  return repl;
}

/// Create a target for use by the repl. The target is created by launching the
/// mojo-repl-entry-point utility executable. The executable is expected to be
/// adjacent to the location of plugin library.
static llvm::Expected<lldb::TargetSP> createMojoReplTarget(Debugger &debugger) {
  // Find the mojo-repl executable. We look for it in the same directory as the
  // plugin.
  FileSpec replEntryPoint(Host::GetModuleFileSpecForHostAddress(
      reinterpret_cast<void *>(createMojoReplTarget)));
  if (!replEntryPoint)
    return createStringError("unable to locate REPL executable");

  replEntryPoint.SetFilename(REPL_ENTRY_POINT_BIN);
  std::string replEntryPointPath(replEntryPoint.GetPath());

  // Make sure the REPL executable exists.
  if (!FileSystem::Instance().Exists(replEntryPoint)) {
    return createStringError("REPL executable does not exist: '{0}'",
                             replEntryPointPath.c_str());
  }

  // Create a target for the repl executable.
  lldb::TargetSP target;
  Status error = debugger.GetTargetList().CreateTarget(
      debugger, replEntryPointPath.c_str(), llvm::sys::getDefaultTargetTriple(),
      eLoadDependentsYes, /*platform_options=*/nullptr, target);
  if (!error.Success()) {
    return createStringError("failed to create REPL target: %s",
                             error.AsCString());
  }
  return target;
}

/// Create a break point within the repl target to provide an anchor for the
/// repl to execute expressions.
static llvm::Error createReplBreakpoint(Target &target) {
  // Limit the breakpoint to the target's executable module.
  lldb::ModuleSP exeModule = target.GetExecutableModule();
  if (!exeModule) {
    target.Destroy();
    return createStringError("unable to resolve REPL executable module");
  }
  FileSpecList containingModules;
  containingModules.Append(exeModule->GetFileSpec());

  // Create the breakpoint.
  lldb::BreakpointSP breakpoint = target.CreateBreakpoint(
      &containingModules, /*containingSourceFiles=*/nullptr,
      /*func_name=*/"main", lldb::eFunctionNameTypeAuto,
      lldb::eLanguageTypeUnknown, /*offset=*/0,
      /*skip_prologue=*/eLazyBoolCalculate, /*internal=*/true,
      /*request_hardware=*/false);
  if (breakpoint->GetNumLocations() == 0)
    return createStringError("failed to resolve REPL breakpoint for 'main'");

  breakpoint->SetBreakpointKind("REPL");
  return llvm::Error::success();
}

/// Launch the repl executable process within the target, and wait for the repl
/// breakpoint to be hit.
llvm::Error MojoREPL::launchEntryPointProcess(Target &target,
                                              Debugger &debugger) {
  // The following disables a warning that is thrown when the entry-point is
  // built with optimizations. This warning pollutes the output and is not
  // helpful because the entry point is actually an empty program.
  ExecutionContext ctx;
  target.CalculateExecutionContext(ctx);
  target.SetPropertyValue(&ctx, lldb_private::eVarSetOperationAssign,
                          /*path=*/"process.optimization-warnings",
                          /*value=*/"false");

  ProcessLaunchInfo launchInfo;
  if (target.GetDisableSTDIO())
    launchInfo.GetFlags().Set(lldb::eLaunchFlagDisableSTDIO);

  lldb::ModuleSP exeModule = target.GetExecutableModule();

  // Configure the launch info to use the target's argv0.
  llvm::StringRef targetSettingsArgv0 = target.GetArg0();
  if (!targetSettingsArgv0.empty()) {
    launchInfo.GetArguments().AppendArgument(targetSettingsArgv0);
    launchInfo.SetExecutableFile(exeModule->GetPlatformFileSpec(), false);
  } else {
    launchInfo.SetExecutableFile(exeModule->GetPlatformFileSpec(), true);
  }

  // Configure the launch environment to use the target's environment. In
  // addition, we also ensure that the library path includes the directory
  // containing the REPL executable.
  launchInfo.GetEnvironment() = target.GetEnvironment();
  launchInfo.GetEnvironment()["LD_LIBRARY_PATH"] +=
      (":" + exeModule->GetFileSpec().GetDirectory().GetStringRef()).str();

  // Launch the process synchronously, waiting for it to stop at the REPL
  // breakpoint.
  debugger.SetAsyncExecution(false);
  Status error = target.Launch(launchInfo, nullptr);
  debugger.SetAsyncExecution(true);
  if (!error.Success()) {
    return createStringError("failed to launch REPL process: {0}",
                             error.AsCString());
  }

  lldb::ProcessSP process = target.GetProcessSP();
  if (!process)
    return createStringError("failed to launch REPL process");

  // Functor used to report an error, and destroy the process.
  auto emitError = [&](StringRef errorMsg) {
    process->Destroy(/*force_kill=*/false);
    return createStringError(errorMsg);
  };

  lldb::StateType state = process->GetState();
  if (state != lldb::eStateStopped)
    return emitError("failed to stop process at REPL breakpoint");

  ThreadList &threadList = process->GetThreadList();
  if (threadList.GetSize() == 0)
    return emitError("process is not in a valid state (no threads)");

  lldb::ThreadSP thread = threadList.GetSelectedThread();
  if (!thread) {
    thread = threadList.GetThreadAtIndex(0);
    threadList.SetSelectedThreadByID(thread->GetID());
    assert(thread && "there should be at least one thread");
  }
  thread->SetSelectedFrameByIndex(0);

  return llvm::Error::success();
}

/// Create a repl instance for a given debugger. `replOptions` contains a set of
/// options to be passed to the repl.
static llvm::Expected<lldb::REPLSP>
createInstanceFromDebugger(Debugger &debugger, const char *replOptions) {
  llvm::Expected<lldb::TargetSP> target = createMojoReplTarget(debugger);
  if (!target)
    return target.takeError();

  // Create a breakpoint in the target to anchor the REPL.
  if (llvm::Error error = createReplBreakpoint(**target))
    return error;

  // Launch the repl process and wait for it to trigger the breakpoint.
  if (llvm::Error error = MojoREPL::launchEntryPointProcess(**target, debugger))
    return error;

  // The process is active and stopped, we can build the REPL now.
  lldb::REPLSP repl = std::make_shared<MojoREPL>(**target);
  repl->SetCompilerOptions(replOptions);
  (*target)->SetREPL(eLanguageTypeMojo, repl);

  if (isatty(STDIN_FILENO))
    printf("Welcome to Mojo.\nType :help for assistance.\n");
  return repl;
}

/// Create a repl instance for either the given target, or the given
/// debuggerer. `replOptions` contains a set of options to be passed to the
/// repl.
static lldb::REPLSP createInstance(Status &error, lldb::LanguageType language,
                                   Debugger *debugger, Target *target,
                                   const char *replOptions) {
  // Needed because the caller might have forgotten to clear this value.
  error.Clear();
  if (target) {
    auto repl = createInstanceFromTarget(*target, replOptions);
    if (repl)
      return *repl;
    return error = repl.takeError(), nullptr;
  }
  if (debugger) {
    auto repl = createInstanceFromDebugger(*debugger, replOptions);
    if (repl)
      return *repl;
    return error = repl.takeError(), nullptr;
  }
  error.SetErrorString("must have a debugger or a target to create a REPL");
  return nullptr;
}

void MojoREPL::Initialize() {
  LanguageSet languages;
  languages.Insert(eLanguageTypeMojo);
  PluginManager::RegisterPlugin(getPluginNameStatic(), "Mojo language REPL",
                                createInstance, languages);
}

void MojoREPL::Terminate() { PluginManager::UnregisterPlugin(createInstance); }

//===----------------------------------------------------------------------===//
// Source Code Handling
//===----------------------------------------------------------------------===//

bool MojoREPL::SourceIsComplete(const std::string &source) {
  SmallVector<StringRef> lines;
  StringRef(source).split(lines, "\n");

  // If the last line is empty, then the source is complete.
  return lines.empty() || lines.back().trim("\r").empty();
}

lldb::offset_t MojoREPL::GetDesiredIndentation(const StringList &lines,
                                               int cursorPosition,
                                               int tabSize) {
  // TODO: Process the input lines to determine the desired indentation.
  return LLDB_INVALID_OFFSET;
}

void MojoREPL::CompleteCode(const std::string &current_code,
                            CompletionRequest &request) {
  // TODO: Implement this when we have code completion functionality in Mojo.
}

//===----------------------------------------------------------------------===//
// Variable Printing
//===----------------------------------------------------------------------===//

bool MojoREPL::PrintOneVariable(Debugger &debugger, lldb::StreamFileSP &output,
                                lldb::ValueObjectSP &valobj,
                                ExpressionVariable *var) {
  // TODO: If a ExpressionVariable was passed, check first if that variable is
  // just an automatically created expression result. These variables are
  // already printed by the REPL so this is done to prevent printing the
  // variable twice.

  valobj->Dump(*output);
  return true;
}
