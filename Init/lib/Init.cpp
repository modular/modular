//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Init/Init.h"
#include "Init/DevelopmentSignalHandler.h"
#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/Configuration.h"
#include "Support/Context.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/Telemetry/Telemetry.h"

#include "llvm/Support/ErrorHandling.h"

#include <cassert>

using namespace M;

namespace {

/// Options used when the global runtime was first created.
AsyncRT::RuntimeOptions &storedGlobalRuntimeCreationOptions() {
  static AsyncRT::RuntimeOptions opts;
  return opts;
}

} // namespace

AsyncRT::RuntimeRef
Init::getOrCreateRuntime(AsyncRT::RuntimeSource source,
                         const AsyncRT::RuntimeOptions &options) {
  std::lock_guard<std::mutex> lock(AsyncRT::getGlobalRuntimeMutex());
  AsyncRT::Runtime *existingRuntime = AsyncRT::getGlobalRuntimePointer();
  if (existingRuntime) {
    AsyncRT::RuntimeOptions &existingOptions =
        storedGlobalRuntimeCreationOptions();
    if (existingOptions != options)
      llvm::report_fatal_error(
          "Init::getOrCreateRuntime called requesting different options to "
          "those used to create the existing Runtime.");
    return AsyncRT::RuntimeRef::copy(existingRuntime);
  }
  AsyncRT::RuntimeRef newRuntime = AsyncRT::createRuntime(source, options);
  storedGlobalRuntimeCreationOptions() = options;
  AsyncRT::setGlobalRuntimePointer(newRuntime.getPointer());
  return newRuntime.copy();
}

static constexpr bool isProductionBuild() {
#ifdef MODULAR_PRODUCTION
  return true;
#else
  return false;
#endif
}

ErrorOr<ContextRef> Init::createContext(StringRef programName,
                                        const Init::Options &options,
                                        StringRef subCommand) {
  // Checks that there is no existing M::Context in the current process, and
  // asserts and returns an error if there is.
  if (getCurrentMaxContextOrNull()) {
    assert(false && "A global context should not already exist.");
    return Error(Twine("A global context should not already exist."));
  }

  // Create the top-level context.
  ContextRef ctx = ContextRef::create();

  // Create the settings object.
  auto settingsOr = Config::open();
  if (settingsOr.isError())
    return settingsOr.takeError();
  Config settings = std::move(*settingsOr);

  bool crashReportingEnabled =
      settings.getValueAsBool("crash_reporting.enabled",
#ifdef MODULAR_PRODUCTION
                              true);
#else
                              false);
#endif // MODULAR_PRODUCTION

  // Enable crash logging, if appropriate.
  if (!isProductionBuild() && !crashReportingEnabled)
    Init::registerDevelopmentSignalHandler(programName);
  else if (!options.forceDisableCrashReporting && crashReportingEnabled)
    initCrashpadForProgram(programName, &settings);

  // Move everything into the context. Construct here may used the settings.
  ctx->emplace<Telemetry::TelemetryContext>(settings, programName, subCommand);

  // Create a new runtime (if needed).
  if (options.runtimeOptions) {
    std::string profileFilename =
        llvm::sys::Process::GetEnv("MODULAR_PROFILE_FILENAME").value_or("");
    AsyncRT::RuntimeOptions opts = *options.runtimeOptions;
    if (!profileFilename.empty())
      opts.profileFilename = profileFilename;
    AsyncRT::RuntimeRef ref =
        Init::getOrCreateRuntime(AsyncRT::RuntimeSource::MaxContext, opts);
    ctx->setRuntime(GenericRCRef::fromRCRef(std::move(ref)));
  }

  // Finally move the settings.
  ctx->emplace<Config>(std::move(settings));

  // Store a copy of the init options so we can compare when reusing the
  // global context.
  ctx->emplace<Init::Options>(options);

  // Set as the global current context so any thread can use
  // getCurrentMaxContext(). We store only a raw pointer; the global does not
  // hold a ref. Cleared in ~Context() when the last ContextRef is destroyed.
  setCurrentMaxContext(ctx.getPointer());

  // Return the useable context.
  return std::move(ctx);
}
