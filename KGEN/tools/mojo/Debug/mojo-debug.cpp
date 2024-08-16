//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-debug.h"
#include "../../common/Telemetry.h"
#include "../Common/CudaGdb.h"
#include "../Common/LLDB.h"
#include "AsyncRT/Init/Init.h"
#include "KGEN/Support/Configuration.h"
#include "RPCServer.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "Debug/DebugOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DebugOptTable : public llvm::opt::PrecomputedOptTable {
  DebugOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Returns the path to the mojo tool, or an error if not found.
static ErrorOr<std::string> getMojoDriver() {
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr))
    return Error(Twine("failed to parse 'modular.cfg': ") +
                 configOr.getError());
  std::error_code ec;
  StringRef driver = configOr->getDriverPath();
  if (!std::filesystem::exists(driver.str(), ec) || ec)
    return Error("unable to resolve the mojo program path");
  return driver.str();
}

static SmallVector<std::string>
getLLDBArgs(llvm::opt::InputArgList &parsedArgs) {
  SmallVector<std::string> lldbArgs;
  for (StringRef value : parsedArgs.getAllArgValues(options::OPT_xlldb))
    lldbArgs.push_back(value.str());
  return lldbArgs;
}

static std::optional<std::string>
getCudaGDBPath(llvm::opt::InputArgList &parsedArgs) {
  StringRef path = parsedArgs.getLastArgValue(options::OPT_cudaGDBPath);
  if (!path.empty())
    return path.str();
  return {};
}

auto getCompilationOptions(llvm::opt::InputArgList &parsedArgs) {
  return parsedArgs.filtered(options::OPT_CompilationOptionGroup,
                             options::OPT_ExperimentalCompilationOptionGroup,
                             options::OPT_DiagnosticOptionGroup);
}

static bool isMojoFile(StringRef file) {
  return file.ends_with(".mojo") || file.ends_with(".🔥");
}

// Given a path, resolve it to an actual full path without dots. If the input is
// a program name, it will be searched in the PATH.
static std::string resolvePath(std::string path) {
  if (!llvm::sys::path::is_absolute(path)) {
    if (std::optional<std::string> fullPath =
            llvm::sys::Process::FindInEnvPath("PATH", path)) {
      path = *fullPath;
    }
  }
  std::error_code ec;
  std::filesystem::path canonicalPath = std::filesystem::canonical(path, ec);
  if (ec)
    return path;
  return canonicalPath.string();
}

/// Launches LLDB with the Mojo plugin enabled, or vanilla cuda-gdb.
/// Exits unsuccessfully if LLDB could not be found in the SDK.
static int debug(const State &state) {
  // Parse command line arguments.
  DebugOptTable options;
  // First, parse all arguments, in order to find the index of the input
  // argument.
  unsigned unused = 0;
  llvm::opt::InputArgList parsedArgs =
      options.ParseArgs(state.arguments, unused, unused);

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext("mojo", Init::Options());
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Initialize telemetry.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), parsedArgs);

  // LLVMOption treats all "positional arguments" (arguments that do not have a
  // "-" or "--" prefix) as `INPUT`. The very first of these is our launch input
  // (a Mojo source file or a binary), and each remaining positional argument is
  // an argument being passed to the debuggee.
  auto positionalArgs = parsedArgs.filtered(options::OPT_INPUT);
  std::optional<std::string> target;
  SmallVector<std::string> runArgs;
  // If we have a positional argument, which is a target, everything that comes
  // after that is a run arg. We then redefine parsedArgs as everything that
  // comes before the target.
  if (!positionalArgs.empty()) {
    const llvm::opt::Arg &targetArg = **positionalArgs.begin();
    target = targetArg.getSpelling();
    runArgs = SmallVector<std::string>(
        state.arguments.slice(targetArg.getIndex() + 1));
    parsedArgs = options.ParseArgs(
        state.arguments.slice(0, targetArg.getIndex()), unused, unused);
  }

  if (parsedArgs.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Debug/DebugOptionsHelpText.inc"
    );
  }

  bool useRpc = parsedArgs.hasArg(options::OPT_rpc);

  std::vector<int> rpcPorts;
  if (parsedArgs.hasArg(options::OPT_port)) {
    StringRef rawRPCPort =
        parsedArgs.getLastArgValue(options::OPT_port, "12346");
    int rpcPort;
    if (rawRPCPort.getAsInteger(10, rpcPort))
      return state.reportError(Twine("invalid RPC port '") + rawRPCPort + "'");
    rpcPorts.push_back(rpcPort);
  } else {
    for (int p = 12355; p <= 12364; ++p)
      rpcPorts.push_back(p);
  }

  bool dryRun = parsedArgs.hasArg(options::OPT_dry_run);
  bool useCudaGDB = parsedArgs.hasArg(options::OPT_cudaGDB);
  std::optional<std::string> cudaGdbPath = getCudaGDBPath(parsedArgs);
  if (cudaGdbPath.has_value() && !useCudaGDB) {
    return state.reportError(Twine("--cuda-gdb-path requires --cuda-gdb"));
  }
  StringRef rpcTerminal =
      parsedArgs.getLastArgValue(options::OPT_terminal, "console");
  SmallVector<std::string> lldbArgs = getLLDBArgs(parsedArgs);
  bool isJITDebugging = target && isMojoFile(*target);
  auto compilationOptions = getCompilationOptions(parsedArgs);

  if (!isJITDebugging && !compilationOptions.empty()) {
    // Compilation options are only allowed when doing JIT debugging.
    return state.reportError(
        Twine("unexpected compilation option '",
              (**compilationOptions.begin()).getSpelling()) +
        "'");
  }

  // This is a launch case.
  if (target) {
    target = resolvePath(*target);

    // This is a JIT debug case, in which case LLDB will debug `mojo run`. We
    // have to include the provided compilation options as run arguments for
    // `mojo run`.
    if (isJITDebugging) {
      ErrorOr<std::string> mojoDriver = getMojoDriver();
      if (failed(mojoDriver))
        return state.reportError(mojoDriver.getError());

      SmallVector<StringRef> mojoRunArgs{"run", "-O0", "--debug-level=full"};
      for (auto arg : compilationOptions) {
        mojoRunArgs.push_back(arg->getSpelling());
        // If at some point we have an option with 2 or more values, we need to
        // figure out a different way to pass the compilation options to the
        // debuggee.
        assert(arg->getNumValues() <= 1);
        for (auto value : arg->getValues())
          mojoRunArgs.push_back(value);
      }
      mojoRunArgs.push_back(*target);
      // `mojo run` args will precede the regular run args, and the actual
      // target will be the mojo driver.
      runArgs.insert(runArgs.begin(), mojoRunArgs.begin(), mojoRunArgs.end());
      target = *mojoDriver;
    }
    if (useRpc) {
      if (useCudaGDB)
        return state.reportError(
            Twine("cuda-gdb with --rpc not yet supported"));
      ErrorOrSuccess status =
          invokeLaunchRPC(dryRun, rpcPorts, *target, runArgs, rpcTerminal);
      if (failed(status))
        return state.reportError(status.getError());
      return 0;
    }

    // When using the LLDB CLI, the first run arg has to be the target.
    runArgs.insert(runArgs.begin(), *target);

    if (useCudaGDB)
      return invokeCudaGdb(state, lldbArgs, runArgs, cudaGdbPath, dryRun);
    else
      return invokeLLDB(state, lldbArgs, runArgs, dryRun);
  }

  std::optional<StringRef> pid;
  if (parsedArgs.hasArg(options::OPT_pid))
    pid = parsedArgs.getLastArgValue(options::OPT_pid);

  std::optional<StringRef> processName;
  if (parsedArgs.hasArg(options::OPT_process_name))
    processName = parsedArgs.getLastArgValue(options::OPT_process_name);

  //  This is an attach case.
  if (pid || processName) {
    if (useRpc) {
      if (useCudaGDB)
        return state.reportError(
            Twine("cuda-gdb with --rpc not yet supported"));
      ErrorOrSuccess status =
          invokeAttachRPC(dryRun, rpcPorts, pid, processName);
      if (failed(status))
        return state.reportError(status.getError());
      return 0;
    } else {
      if (pid)
        llvm::append_values(lldbArgs, std::string("-p"), pid->str());
      if (processName)
        llvm::append_values(lldbArgs, std::string("-n"),
                            resolvePath(processName->str()));
      if (useCudaGDB)
        return invokeCudaGdb(state, lldbArgs, {}, cudaGdbPath, dryRun);
      else
        return invokeLLDB(state, lldbArgs, {}, dryRun);
    }
  }

  for (auto rpcArg : parsedArgs.filtered(options::OPT_RPCOptionGroup))
    return state.reportError(
        Twine("unexpected option '", rpcArg->getSpelling()) + "'");

  // This is a regular cli passthrough.
  if (useCudaGDB)
    return invokeCudaGdb(state, lldbArgs, {}, cudaGdbPath, dryRun);
  else
    return invokeLLDB(state, lldbArgs, {}, dryRun);
}

void M::registerDebugSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("debug", debug);
}
