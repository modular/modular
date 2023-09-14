//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLDB.h"
#include "../Common/Telemetry.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLDB/MojoLLDB.h"
#include "Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include <filesystem>

using namespace M;

llvm::ErrorOr<std::string> M::getLLDB(const std::string &executable) {
  // Attempt to find an LLDB installed alongside the driver.
  std::string str = std::filesystem::path(executable).parent_path().string();
  return llvm::sys::findProgramByName("lldb",
                                      /*Paths=*/ArrayRef<StringRef>(str));
}

/// Returns the path to the MojoLLDB shared library, or an error if not found.
/// This library implements Mojo's LLDB plugin.
static ErrorOr<std::string> getMojoLLDB() {
  // Read the mojo configuration.
  ErrorOr<Config> configOr = Config::open();
  if (failed(configOr)) {
    return Error(Twine("failed to parse 'modular.cfg': ") +
                 configOr.getError());
  }
  Config config = std::move(*configOr);

  std::error_code ec;
  StringRef mojoLLDB = config.getValue("mojo.lldb_plugin_path");
  if (!std::filesystem::exists(mojoLLDB.str(), ec) || ec)
    return Error("unable to resolve the MojoLLDB plugin path");
  return mojoLLDB.str();
}

int M::invokeLLDB(const State &state, llvm::opt::InputArgList &args,
                  std::initializer_list<StringRef> extraOptions) {
  // Initialize the LLCL runtime. We don't allow users to configure runtime
  // options, such as the allocator or the work queue threading model.
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

  // Initialize telemetry.
  auto &telemetryCtx = runtime.emplaceContext<M::Telemetry::TelemetryContext>();
  initializeTelemetry(telemetryCtx, state, args);

  // Find the path to the LLDB executable and the MojoLLDB plugin library.
  std::string executable =
      llvm::sys::fs::getMainExecutable(state.programName, (void *)M::getLLDB);
  llvm::ErrorOr<std::string> lldb = M::getLLDB(executable);
  if (!lldb)
    return state.reportError("lldb must be installed alongside mojo");
  ErrorOr<std::string> mojoLLDB = getMojoLLDB();
  if (failed(mojoLLDB))
    return state.reportError(mojoLLDB.getError());

  // We forward all unparsed command line arguments to LLDB.
  SmallVector<StringRef> lldbArgs(state.arguments);
  std::string loadCommand = llvm::formatv("plugin load \"{0}\"", *mojoLLDB);
  lldbArgs.insert(lldbArgs.begin(),
                  {lldb.get(), "-Q", "--one-line-before-file", loadCommand});
  lldbArgs.insert(lldbArgs.end(), extraOptions);
  return llvm::sys::ExecuteAndWait(lldb.get(), lldbArgs);
}
