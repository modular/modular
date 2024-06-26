//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CrashReporting/CrashReporting.h"

#include "Config/Version.h"
#include "Support/ErrorOr.h"
#include "Support/Settings/Settings.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "client/crash_report_database.h"
#include "client/crashpad_client.h"
#include "client/settings.h"
#include "client/simulate_crash.h"

using namespace M;

static constexpr llvm::StringLiteral kHandlerProgramName =
    "modular-crashpad-handler";
static constexpr llvm::StringLiteral kDefaultURL =
    "https://crash-reporting.modular.com";

std::filesystem::path
M::getCrashDatabasePath(Settings *settings,
                        const std::filesystem::path &dataFolder) {
  if (settings) {
    auto *setting = settings->get("crash_reporting.database_path");
    auto path = llvm::dyn_cast_if_present<StringRef>(setting);
    if (!path.empty())
      return std::string_view(path);
  }
  return dataFolder / "crashdb";
}

ErrorOr<std::filesystem::path> M::getCrashpadHandlerPath(Settings *settings) {
  StringRef program("");
  if (settings) {
    auto *handlerPath = settings->get("crash_reporting.handler_path");
    program = llvm::dyn_cast_if_present<StringRef>(handlerPath);
  }
  std::string foundProgram;
  if (program.empty()) {
    // N.B.: Errors from findProgramByName are intentionally ignored for the
    // same reason as above.
    if (auto programOr = llvm::sys::findProgramByName(kHandlerProgramName)) {
      foundProgram = std::move(*programOr);
      program = foundProgram;
    }
  }
  // No luck.
  if (program.empty())
    return Error("unable to locate crashpad handler executable");
  return std::string_view(program);
}

static ErrorOrSuccess tryInitCrashpad(StringRef program, Settings *settings) {
  if (settings) {
    bool enabled = settings->getBool("crash_reporting.enabled", true);
    if (!enabled)
      return success();
  }

  // Crashpad needs a few paths and other configuration bits:
  //   - Path of the handler executable (This runs alongside the Mojo driver;
  //     in case the driver crashes, the handler inspects the driver in its
  //     crashed state and generates a crash report)
  //   - Crash database, to put the crashes in before they are sent off
  //   - URL to upload crash reports to
  auto dataFolderOr = Config::getModularDataFolderPath();
  if (dataFolderOr.isError())
    return dataFolderOr.takeError();
  std::filesystem::path databasePath =
      getCrashDatabasePath(settings, *dataFolderOr);

  auto handlerPathOr = getCrashpadHandlerPath(settings);
  if (handlerPathOr)
    return Error(llvm::Twine("while locating crashpad handler: ") +
                 handlerPathOr.getError());
  std::filesystem::path handlerPath = std::move(*handlerPathOr);
  StringRef url("");
  if (settings) {
    auto *urlSetting = settings->get("crash_reporting.url");
    url = llvm::dyn_cast_if_present<StringRef>(urlSetting);
  }
  std::string defaultURL;

  // If the URL is empty, construct a URL by appending to the default URL
  // above. This is one way to communicate a high-level categorization without
  // having to dissemble the minidump on the server-side. However, most
  // attributes should go into the attribute map below.
  if (url.empty()) {
    defaultURL = std::string(kDefaultURL) + "/" + std::string(program);
    url = defaultURL;
  }

  // Update the database if we have a URL and reporting is not enabled. In most
  // cases this will just read the existing database settings and not change.
  if (!url.empty()) {
    auto database =
        crashpad::CrashReportDatabase::Initialize(base::FilePath(databasePath));
    bool uploads_enabled = false;
    if (database != nullptr && database->GetSettings() != nullptr &&
        (!database->GetSettings()->GetUploadsEnabled(&uploads_enabled) ||
         !uploads_enabled))
      database->GetSettings()->SetUploadsEnabled(true);
  }

  // Setup all the annotations.
  std::map<std::string, std::string> annotations;
  annotations["program"] = std::string(program);
  annotations["version"] = getModularVersionString();

  // Launch Crashpad handler.
  crashpad::CrashpadClient client;
  if (!client.StartHandler(
          base::FilePath(handlerPath), base::FilePath(databasePath),
          /*metrics_dir=*/base::FilePath(databasePath), std::string(url),
          /*annotations=*/annotations,
          /*arguments=*/{}, /*restartable=*/true,
          /*asynchronous_start=*/false))
    return Error("crashpad failed to start handler");
  return success();
}

void M::initCrashpadForProgram(StringRef program, Settings *settings) {
  if (auto error = tryInitCrashpad(program, settings))
    llvm::errs() << "Failed to initialize Crashpad.  "
                    "Crash reporting will not be available.  "
                    "Cause: "
                 << error.getError() << "\n";
}

void M::generateNonFatalDump() { CRASHPAD_SIMULATE_CRASH(); }
