//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CrashReporting.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "client/crash_report_database.h"
#include "client/crashpad_client.h"
#include "client/settings.h"
#include "client/simulate_crash.h"

using namespace M;

static constexpr llvm::StringLiteral kHandlerProgramName =
    "modular-crashpad-handler";
// TODO(#18360): Add a default crashpad upload URL here
static constexpr llvm::StringLiteral kDefaultUploadURL = "";

std::filesystem::path
M::getCrashDatabasePath(Config &config,
                        const std::filesystem::path &modularHome) {
  StringRef fromConfig = config.getValue("crash_reporting.database_path");
  if (!fromConfig.empty())
    return std::string_view(fromConfig);
  return modularHome / "crashdb";
}

ErrorOr<std::filesystem::path> M::getCrashpadHandlerPath(Config &config,
                                                         const char *argv0) {
  // Logic similar to printSymbolizedStackTrace inside LLVM.
  // Highest precedence: configuration value.
  // Note: Can't use StringRef here because need to keep value alive in
  // findProgramByName cases.
  std::string program = config.getValue("crash_reporting.handler_path").str();
  // Next best: Handler living alongside current executable as reported by
  // argv[0]
  if (program.empty()) {
    StringRef parent = llvm::sys::path::parent_path(argv0);
    if (!parent.empty()) {
      // N.B.: Errors from findProgramByName are intentionally ignored.
      // At least on Unix, the only error it ever returns is "file not found".
      // Such an error should not prevent attempts of further alternatives.
      if (auto programOr =
              llvm::sys::findProgramByName(kHandlerProgramName, parent))
        program = std::move(*programOr);
    }
  }
  // Next best (not part of printSymbolizedStackTrace): Handler living
  // alongside current executable as reported by getMainExecutable.  This is not
  // a part of the printSymbolizedStackTrace, but is necessary in the mojo-lldb
  // case.
  if (program.empty()) {
    std::string mainExecutable = llvm::sys::fs::getMainExecutable(
        argv0, reinterpret_cast<void *>(initCrashpadForProgram));
    StringRef parent = llvm::sys::path::parent_path(mainExecutable);
    if (!parent.empty()) {
      // N.B.: Errors from findProgramByName are intentionally ignored for the
      // same reason as above.
      if (auto programOr =
              llvm::sys::findProgramByName(kHandlerProgramName, parent))
        program = std::move(*programOr);
    }
  }
  // Next best: Handler anywhere on the path.
  if (program.empty()) {
    // N.B.: Errors from findProgramByName are intentionally ignored for the
    // same reason as above.
    if (auto programOr = llvm::sys::findProgramByName(kHandlerProgramName))
      program = std::move(*programOr);
  }
  // No luck.
  if (program.empty())
    return Error("unable to locate crashpad handler executable");
  return std::string_view(program);
}

/// Attempt to initialize Crashpad (returning an error upon failure).
static ErrorOrSuccess tryInitCrashpad(const char *argv0) {
  auto configOr = Config::open();
  if (configOr)
    return Error(llvm::Twine("while reading configuration: ") +
                 configOr.getError());
  auto config = std::move(*configOr);
  if (!config.getValue("crash_reporting.disabled").empty())
    return success();

  // Crashpad needs a few paths and other configuration bits:
  //   - Path of the handler executable (This runs alongside the Mojo driver;
  //     in case the driver crashes, the handler inspects the driver in its
  //     crashed state and generates a crash report)
  //   - Crash database, to put the crashes in before they are sent off
  //   - URL to upload crash reports to
  std::filesystem::path modularHome = Config::getModularHomeDirPath();
  std::filesystem::path databasePath =
      getCrashDatabasePath(config, modularHome);
  auto handlerPathOr = getCrashpadHandlerPath(config, argv0);
  if (handlerPathOr)
    return Error(llvm::Twine("while locating crashpad handler: ") +
                 handlerPathOr.getError());
  std::filesystem::path handlerPath = std::move(*handlerPathOr);
  StringRef url = config.getValue("crash_reporting.url");
  if (url.empty())
    url = kDefaultUploadURL;

  // Launch Crashpad handler.
  crashpad::CrashpadClient client;
  if (!client.StartHandler(
          base::FilePath(handlerPath), base::FilePath(databasePath),
          /*metrics_dir=*/base::FilePath(databasePath), std::string(url),
          /*annotations=*/{},
          /*arguments=*/{}, /*restartable=*/true,
          /*asynchronous_start=*/false))
    return Error("crashpad failed to start handler");
  return success();
}

void M::initCrashpadForProgram(const char *argv0) {
  if (auto error = tryInitCrashpad(argv0))
    llvm::errs() << "Failed to initialize Crashpad.  "
                    "Crash reporting will not be available.  "
                    "Cause: "
                 << error.getError() << "\n";
}

void M::generateNonFatalDump() { CRASHPAD_SIMULATE_CRASH(); }
