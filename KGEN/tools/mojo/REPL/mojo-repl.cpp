//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-repl.h"
#include "../Common/Telemetry.h"
#include "LLCL/Runtime/Runtime.h"
#include "REPL/MojoLLDB.h"

#include "Support/Driver/DriverSupport.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

#include <filesystem>

using namespace M;

#define DRIVER_OPTIONS_PATH "REPL/REPLOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct REPLOptTable : public llvm::opt::PrecomputedOptTable {
  REPLOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Returns the path to a suitable `lldb` executable that can be used to launch
/// the REPL, or an error if none exists.
static llvm::ErrorOr<std::string> getLLDB(const std::string &executable) {
  // Attempt to find an lldb installed alongside the driver.
  std::string str = std::filesystem::path(executable).parent_path().string();
  return llvm::sys::findProgramByName("lldb",
                                      /*Paths=*/ArrayRef<StringRef>(str));
}

/// Returns the path to a MojoLLDB dynamic library, or an error if none exists.
/// This library implements Mojo's lldb plugin.
static std::string getMojoLLDB(const std::string &executable) {
  // Attempt to find a MojoLLDB installed relative to the driver: if the driver
  // exists at "foo/bin/mojo", MojoLLDB should exist at "foo/bin/../lib/".
  std::filesystem::path lib =
      std::filesystem::path(executable).parent_path().parent_path() / "lib" /
      MOJO_LLDB;
  return lib.string();
}

/// Launches the Mojo REPL, which is in fact an invocation of
/// `lldb --repl-language mojo`. Exits unsuccessfully if lldb could not be found
/// in the user's PATH.
static int repl(const State &state) {
  // Parse command line arguments. We forward most arguments to the underlying
  // invocation of lldb, and so don't check for invalid options.
  REPLOptTable options;
  unsigned unused = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, unused, unused);

  if (args.hasArg(options::OPT_help, options::OPT_help_text)) {
    return state.printHelp(/*plainText=*/args.hasArg(options::OPT_help_text),
#include "REPL/REPLOptionsHelpText.inc"
    );
  }

  // Initialize the LLCL runtime. We don't allow users to configure runtime
  // options, such as the allocator or the work queue threading model.
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

  // Initialize telemetry.
  auto &telemetryCtx = runtime.emplaceContext<M::Telemetry::TelemetryContext>();
  initializeTelemetry(telemetryCtx, state, args);

  // Find the path to the lldb executable and the MojoLLDB plugin library.
  std::string executable =
      llvm::sys::fs::getMainExecutable(state.programName, (void *)getLLDB);
  llvm::ErrorOr<std::string> lldb = getLLDB(executable);
  if (!lldb)
    return state.reportError(
        "lldb must be installed alongside mojo in order to launch the REPL");
  std::string mojoLLDB = getMojoLLDB(executable);

  // We forward all unparsed command line arguments to lldb, as values for the
  // `--repl` option.
  SmallVector<StringRef> lldbArgs(state.arguments);
  std::string loadCommand = llvm::formatv("plugin load \"{0}\"", mojoLLDB);
  lldbArgs.insert(lldbArgs.begin(),
                  {lldb.get(), "-Q", "--one-line-before-file", loadCommand,
                   "--one-line-before-file", "settings set show-progress false",
                   "--repl-language", "mojo", "--repl"});
  return llvm::sys::ExecuteAndWait(lldb.get(), lldbArgs);
}

void M::registerREPLSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("repl", repl);
}
