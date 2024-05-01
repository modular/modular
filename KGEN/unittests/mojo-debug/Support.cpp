//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "KGEN/Support/Configuration.h"
#include "lldb/API/SBDebugger.h"
#include "llvm/Support/Program.h"
#include "gtest/gtest.h"

using namespace M;
using namespace lldb;

/// The leak sanitizer shows errors, probably because we load libpython via
/// LLDB.
extern "C" const char *__asan_default_options() { return "detect_leaks=0"; }

static TempDir createTempDir() {
  ErrorOr<TempDir> tempDirOr = TempDir::create("mojo-debug.%%%%%%");
  if (failed(tempDirOr))
    llvm::report_fatal_error(tempDirOr.takeError().get());
  return std::move(*tempDirOr);
}

MojoSource::MojoSource(StringRef fileName) {
  path = std::filesystem::path(std::getenv("MODULAR_PATH")) / "KGEN" /
         "unittests" / "mojo-debug" / "inputs" / fileName.str();
  pathStr = path.string();

  auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(pathStr));
  if (failed(bufferOr))
    llvm::report_fatal_error(Twine("Error reading the file ") + pathStr + ": " +
                             bufferOr.getError());
  llvm::MemoryBuffer &buffer = *bufferOr->get();
  contents = buffer.getBuffer();

  StringRef(contents).split(lines, '\n');
}

std::vector<int> MojoSource::findLinesWithText(StringRef text) const {
  std::vector<int> result;
  for (size_t i = 0, e = lines.size(); i < e; ++i)
    if (lines[i].contains(text))
      result.push_back(i + 1);
  return result;
}

MojoBinary::MojoBinary(const std::shared_ptr<MojoSource> &source,
                       bool suppressBuildOutput)
    : source(source), outDir(createTempDir()),
      binPath(outDir.getPath() /
              (source->getFilesystemPath().filename().string() + ".exe")) {
  ErrorOr<std::string> mojoOr =
      toModularErrorOr(llvm::sys::findProgramByName("mojo"));
  if (failed(mojoOr))
    llvm::report_fatal_error(mojoOr.getError());

  std::vector<std::optional<StringRef>> redirects;
  if (suppressBuildOutput) {
    for (size_t i = 0; i < 3; ++i)
      redirects.emplace_back("");
  }
  int ec = llvm::sys::ExecuteAndWait(
      *mojoOr,
      {*mojoOr, "build", "-g", "-O0", source->getPath(), "-o", binPath},
      /*Env=*/std::nullopt, redirects);
  if (ec)
    llvm::report_fatal_error(llvm::Twine("mojo build exit code = ") +
                             std::to_string(ec));
}

/// Acquire a singleton instance of a debugger.
static SBDebugger getOrCreateSBDebugger() {
  static std::once_flag flag;
  static SBDebugger debugger;
  std::call_once(flag, []() {
    // Initialize the singleton debugger.
    SBError err = SBDebugger::InitializeWithErrorHandling();
    ASSERT_FALSE(err.Fail()) << err.GetCString();
    debugger = SBDebugger::Create(/*source_init_files=*/false);
    debugger.SetAsync(false);

    // Launch the test lldbinit file
    SBFileSpec lldbInitPath(
        (std::filesystem::path(std::getenv("MODULAR_PATH")) / "utils" /
         "lit-lldb-init.in")
            .string()
            .c_str());

    SBCommandReturnObject result;
    SBExecutionContext exeCtx;
    SBCommandInterpreterRunOptions options;
    options.SetPrintResults(false);
    options.SetEchoCommands(false);
    debugger.GetCommandInterpreter().HandleCommandsFromFile(
        lldbInitPath, exeCtx, options, result);

    if (std::string error = result.GetError(); !error.empty())
      llvm::outs() << std::string(result.GetOutput()) << "\n" << error << "\n";

    // Load the MojoLLDB plugin
    ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
    if (failed(configOr))
      llvm::report_fatal_error(Twine("failed to parse 'modular.cfg': ") +
                               configOr.getError());
    std::error_code ec;
    StringRef mojoLLDB = configOr->getLLDBPluginPath();
    if (!std::filesystem::exists(mojoLLDB.str(), ec) || ec)
      llvm::report_fatal_error("unable to resolve the MojoLLDB plugin path");
    debugger.HandleCommand(("plugin load " + mojoLLDB).str().c_str());
  });
  return debugger;
}

/// Execute the provided command using the provided context (thread, process or
/// frame).
///
/// Note: it's better to use this instead of `debugger.HandleCommand()` because
/// it doesn't work nicely if multiple targets exist at once, which happens when
/// multiple test files are executed simultaneously.
template <typename Ctx>
static CommandResult runCommandForContext(StringRef command, Ctx context) {
  SBCommandReturnObject result;
  SBExecutionContext exeCtx(context);
  getOrCreateSBDebugger().GetCommandInterpreter().HandleCommand(command.data(),
                                                                exeCtx, result);

  std::string output = std::string(result.GetOutput());
  std::string error = std::string(result.GetError());
  return {result.Succeeded(), output, error};
}

/// Similar to runCommandForContext, but the output and error are printed
/// right away.
///
/// Returns true if and only if the command succeeded.
template <typename Ctx>
static bool dumpCommandForContext(StringRef command, Ctx context) {
  CommandResult result = runCommandForContext(command, context);
  if (!result.output.empty())
    llvm::outs() << result.output << "\n";
  if (!result.error.empty())
    llvm::outs() << result.error << "\n";
  return result.success;
}

/// Traverses the input file looking for the `# breakpoint` comment, and
/// places a breakpoint at the lines where it appears.
static void setBreakpointsForComments(const MojoSource &source,
                                      SBTarget &target) {

  for (int line : source.findLinesWithText("# breakpoint")) {
    SBBreakpoint bp =
        target.BreakpointCreateByLocation(source.getPath().data(), line);
    if (bp.GetNumLocations() != 1)
      llvm::report_fatal_error(llvm::formatv(
          "Couldn't set a breakpoint at {0}:{1}", source.getPath(), line));
  }
}

static StopContext runTarget(SBTarget target, MojoBinary binary) {
  // We use this command because it nicely uses all the defaults from
  // the lldb init file, unlike debugger.Launch.
  if (std::getenv("DUMP_STOP_CONTEXT_AT_LAUNCH")) {
    dumpCommandForContext("run", target);
  } else {
    const char **argv = {};
    target.LaunchSimple(argv, nullptr, nullptr);
  }

  SBProcess process = target.GetProcess();
  if (!process.IsValid())
    llvm::report_fatal_error("Invalid process");

  // This ensures the process didn't exit
  if (process.GetState() != lldb::eStateStopped)
    llvm::report_fatal_error("Process is not stopped");

  SBThread thread = process.GetSelectedThread();
  return StopContext{std::move(binary), target, process, thread,
                     thread.GetFrameAtIndex(0)};
}

StopContext StopContext::stepOver() {
  thread.StepOver();

  if (process.GetState() != lldb::eStateStopped)
    llvm::report_fatal_error("Process is not stopped after step over");

  SBThread newThread = process.GetSelectedThread();

  return StopContext{std::move(binary), target, process, newThread,
                     newThread.GetFrameAtIndex(0)};
}

StopContext StopContext::stepInto() {
  thread.StepInto();

  if (process.GetState() != lldb::eStateStopped)
    llvm::report_fatal_error("Process is not stopped after step over");

  SBThread newThread = process.GetSelectedThread();

  return StopContext{std::move(binary), target, process, newThread,
                     newThread.GetFrameAtIndex(0)};
}

StopContext StopContext::resume() {
  process.Continue();

  if (process.GetState() != lldb::eStateStopped)
    llvm::report_fatal_error("Process is not stopped after step over");

  SBThread newThread = process.GetSelectedThread();

  return StopContext{std::move(binary), target, process, newThread,
                     newThread.GetFrameAtIndex(0)};
}

CommandResult StopContext::runCommand(StringRef command) {
  return runCommandForContext(command, frame);
}

StopContext M::buildAndLaunch(StringRef fileName) {
  auto source = std::make_shared<MojoSource>(fileName);

  // TODO(28608): support a test mode for JIT debugging besides AOT.
  MojoBinary binary(source, /*suppressBuildOutput=*/true);
  SBTarget target =
      getOrCreateSBDebugger().CreateTarget(binary.getPath().data());
  if (!target.IsValid())
    llvm::report_fatal_error("Invalid target");

  setBreakpointsForComments(*source, target);

  return runTarget(target, std::move(binary));
}
