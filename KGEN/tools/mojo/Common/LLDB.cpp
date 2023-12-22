//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLDB.h"
#include "../Common/Telemetry.h"
#include "Debug/MojoDebug.h"
#include "KGEN/Support/Configuration.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include <filesystem>

#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace M;

/// Returns the path to the `lldb` executable, or an error if not found.
static ErrorOr<std::string> getLLDB(KGEN::MojoConfig &config) {
  std::error_code ec;
  StringRef lldb = config.getLLDBPath();
  if (!std::filesystem::exists(lldb.str(), ec) || ec)
    return Error("unable to resolve the lldb path");
  return lldb.str();
}

/// Returns the path to the MojoLLDB shared library, or an error if not found.
/// This library implements Mojo's LLDB plugin.
static ErrorOr<std::string> getMojoLLDB(KGEN::MojoConfig &config) {
  std::error_code ec;
  StringRef mojoLLDB = config.getLLDBPluginPath();
  if (!std::filesystem::exists(mojoLLDB.str(), ec) || ec)
    return Error("unable to resolve the MojoLLDB plugin path");
  return mojoLLDB.str();
}

int M::invokeLLDB(const State &state, llvm::opt::InputArgList &args,
                  std::initializer_list<StringRef> extraOptions) {
  // Initialize the LLCL runtime. We don't allow users to configure runtime
  // options, such as the allocator or the work queue threading model.
  std::unique_ptr<LLCL::Runtime> runtime = LLCL::createRuntime();

  // Initialize telemetry.
  auto &telemetryCtx =
      runtime->emplaceContext<M::Telemetry::TelemetryContext>();
  initializeTelemetry(telemetryCtx, state, args);

  // Find the path to the LLDB executable and the MojoLLDB plugin library.
  // Read the mojo configuration.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr)) {
    return state.reportError(Twine("failed to parse 'modular.cfg': ") +
                             configOr.getError());
  }

  KGEN::MojoConfig config = std::move(*configOr);
  ErrorOr<std::string> lldb = getLLDB(config);
  if (failed(lldb))
    return state.reportError(lldb.getError());
  ErrorOr<std::string> mojoLLDB = getMojoLLDB(config);
  if (failed(mojoLLDB))
    return state.reportError(mojoLLDB.getError());

  // We forward all unparsed command line arguments to LLDB.
  SmallVector<StringRef> lldbArgs(state.arguments);
  std::string loadCommand = llvm::formatv("plugin load \"{0}\"", *mojoLLDB);
  lldbArgs.insert(lldbArgs.begin(),
                  {lldb.get(), "-Q", "--one-line-before-file", loadCommand});
  lldbArgs.insert(lldbArgs.end(), extraOptions);

  // We use execv to ensure LLDB replaces the same process so that we can more
  // easily debug it with `lldb -- mojo debug` or `lldb -- mojo repl`.
  size_t size = lldbArgs.size();
  char **execvArgs = new char *[size + 1];
  for (size_t i = 0; i < size; ++i)
    execvArgs[i] = const_cast<char *>(lldbArgs[i].data());
  execvArgs[size] = nullptr;
#if defined(_WIN32)
  return llvm::sys::ExecuteAndWait(lldb.get(), lldbArgs);
#else
  return execv(lldb.get().c_str(), execvArgs);
#endif
}
