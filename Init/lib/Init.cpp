//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Init/Init.h"
#include "Config/Version.h"
#include "Init/DevelopmentSignalHandler.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/Configuration.h"
#include "Support/Context.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/HTTP/HTTPClient.h"
#include "Support/Telemetry/Telemetry.h"

using namespace M;

static constexpr bool isProductionBuild() {
#ifdef MODULAR_PRODUCTION
  return true;
#else
  return false;
#endif
}

ErrorOr<ContextRef> Init::createContext(StringRef programName,
                                        const Init::Options &options) {
  // Create the top-level context.
  ContextRef ctx = ContextRef::create();

  // Create our global HTTP context.
  auto httpCtx = HTTPContext::init();

  // Set basic details on the context, including our version.
  httpCtx->setUserAgent("modular-" + std::string(programName) + "/" +
                        std::string(getModularVersionString()));

  // Create the settings object.
  auto settingsOr = Config::open();
  if (settingsOr.isError())
    return settingsOr.takeError();
  Config settings = std::move(*settingsOr);

  // If we have a certificate authority, set that on the HTTPContext.
  StringRef caInfo = settings.getValue("ssl.cainfo");
  if (!caInfo.empty())
    httpCtx->setCAInfo(std::string(caInfo));

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
  ctx->emplace<HTTPContextRef>(std::move(httpCtx));
  ctx->emplace<Telemetry::TelemetryContext>(settings);

  // Create a new runtime (if needed).
  if (options.runtimeOptions) {
    std::string profileFilename =
        llvm::sys::Process::GetEnv("MODULAR_PROFILE_FILENAME").value_or("");

    AsyncRT::CompactRuntimePtr runtimePtr =
        AsyncRT::CompactRuntimePtr::reserve();
    std::unique_ptr<Allocator> allocator =
        AsyncRT::getAllocator(options.runtimeOptions->getAllocatorOptions());
    std::unique_ptr<AsyncRT::WorkQueue> workQueue =
        options.runtimeOptions->singleThreaded
            ? AsyncRT::createSingleThreadWorkQueue(runtimePtr)
            : AsyncRT::createThreadPoolWorkQueue(
                  runtimePtr, options.runtimeOptions->numThreads,
                  options.runtimeOptions->maxThreads,
                  options.runtimeOptions->mainWillDonate,
                  options.runtimeOptions->withAffinity,
                  std::chrono::microseconds(
                      options.runtimeOptions->threadBusyWaitTime),
                  options.runtimeOptions->poolName);
    ctx->emplace<Runtime>(runtimePtr, ctx.getPointer(), std::move(allocator),
                          std::move(workQueue),
                          profileFilename.empty()
                              ? options.runtimeOptions->profileFilename
                              : profileFilename,
                          options.runtimeOptions->runtimeProfilingTypeMask,
                          options.runtimeOptions->profilerDebuginfo);
  }

  // Finally move the settings.
  ctx->emplace<Config>(std::move(settings));

  // Return the useable context.
  return std::move(ctx);
}
