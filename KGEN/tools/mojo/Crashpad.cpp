//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Crashpad.h"
#include "Support/Configuration.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "client/crash_report_database.h"
#include "client/crashpad_client.h"
#include "client/settings.h"

using namespace M;

static constexpr llvm::StringLiteral kHandlerProgramName =
    "modular-crashpad-handler";
// TODO(#18360): Add a default crashpad upload URL here
static constexpr llvm::StringLiteral kDefaultUploadURL = "";

/// Pick a location to store crash data in.
///
/// Prefers a value from the "crash_reporting.database_path" configuration
/// option, but will fall back to a "crashdb" directory inside of the modular
/// home directory.
static std::filesystem::path
getDatabasePath(Config &config, const std::filesystem::path &modularHome) {
  StringRef fromConfig = config.getValue("crash_reporting.database_path");
  if (!fromConfig.empty())
    return std::string_view(fromConfig);
  return modularHome / "crashdb";
}

/// Convert an LLVM ErrorOr value to a Modular ErrorOr value.
template <typename T>
static ErrorOr<T> toModularError(llvm::ErrorOr<T> llvmErrorOr) {
  // Note: llvm::ErrorOr's operator bool is inverted from Modular's, so this
  // sees whether it was successful, not whether it failed
  if (llvmErrorOr)
    return std::move(*llvmErrorOr);
  return Error(llvmErrorOr.getError().message());
}

/// Attempt to locate the Crashpad handler executable.
///
/// If specified in the configuration, that takes precedence.  Otherwise, we
/// look alongside the running executable, or failing that, anywhere on the
/// PATH.
static ErrorOr<std::filesystem::path> getHandlerPath(Config &config,
                                                     const char *argv0) {
  // Logic similar to printSymbolizedStackTrace inside LLVM
  // Highest precedence: configuration value
  // Note: Can't use StringRef here because need to keep value alive in
  // findProgramByName cases
  std::string program = config.getValue("crash_reporting.handler_path").str();
  // Next best: Handler living alongside current executable
  if (program.empty()) {
    StringRef parent = llvm::sys::path::parent_path(argv0);
    if (!parent.empty())
      UNWRAP_ERROR_OR_SET(program, toModularError(llvm::sys::findProgramByName(
                                       kHandlerProgramName, parent)));
  }
  // Next best: Handler anywhere on the path
  if (program.empty())
    UNWRAP_ERROR_OR_SET(program, toModularError(llvm::sys::findProgramByName(
                                     kHandlerProgramName)));
  // No luck
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
  std::filesystem::path databasePath = getDatabasePath(config, modularHome);
  auto handlerPathOr = getHandlerPath(config, argv0);
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

void M::initCrashpad(const char *argv0) {
  if (auto error = tryInitCrashpad(argv0))
    llvm::errs() << "Failed to initialize Crashpad.  "
                    "Crash reporting will not be available.  "
                    "Cause: "
                 << error.getError() << "\n";
}
