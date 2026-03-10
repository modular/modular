//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Init/Init.h"
#include "Init/DevelopmentSignalHandler.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/Configuration.h"
#include "Support/Context.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/Telemetry/Telemetry.h"

using namespace M;

namespace {

bool runtimeOptionsEqualIgnoringPoolName(const AsyncRT::RuntimeOptions &a,
                                         const AsyncRT::RuntimeOptions &b) {
  return a.numThreads == b.numThreads && a.maxThreads == b.maxThreads &&
         a.singleThreaded == b.singleThreaded &&
         a.profileFilename == b.profileFilename &&
         a.runtimeProfilingTypeMask == b.runtimeProfilingTypeMask &&
         a.mainWillDonate == b.mainWillDonate &&
         a.threadBusyWaitTime == b.threadBusyWaitTime &&
         a.withAffinity == b.withAffinity &&
         a.leakCheckedAllocator == b.leakCheckedAllocator &&
         a.tcmallocAllocator == b.tcmallocAllocator &&
         a.profilingAllocator == b.profilingAllocator &&
         a.useAfterFreeAllocator == b.useAfterFreeAllocator &&
         a.onFailure == b.onFailure && a.workQueueType == b.workQueueType &&
         a.allocatorType == b.allocatorType &&
         a.profilerDebuginfo == b.profilerDebuginfo &&
         a.defaultWorkQueue == b.defaultWorkQueue;
}

} // namespace

bool Init::optionsEqualIgnoringPoolName(const Options &a, const Options &b) {
  if (a.forceDisableCrashReporting != b.forceDisableCrashReporting)
    return false;
  if (a.runtimeOptions.has_value() != b.runtimeOptions.has_value())
    return false;
  if (a.runtimeOptions.has_value())
    return runtimeOptionsEqualIgnoringPoolName(*a.runtimeOptions,
                                               *b.runtimeOptions);
  return true;
}

static constexpr bool isProductionBuild() {
#ifdef MODULAR_PRODUCTION
  return true;
#else
  return false;
#endif
}

ErrorOr<ContextRef> Init::createContext(StringRef programName,
                                        const Init::Options &options) {
  // If a global context already exists (e.g. second InferenceSession), return
  // a ref to it instead of creating a second context.
  Context *existing = getCurrentMaxContextOrNull();
  if (existing) {
    Init::Options *existingOptions = existing->get<Init::Options>();
    assert(existingOptions &&
           "Existing Max context has no Init::Options (cannot compare).");
    assert(Init::optionsEqualIgnoringPoolName(*existingOptions, options) &&
           "Existing Max context was created with different Init::Options.");
    return getCurrentMaxContext();
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
    ctx->emplace<Runtime>(
        runtimePtr, std::move(allocator), std::move(workQueue),
        profileFilename.empty() ? options.runtimeOptions->profileFilename
                                : profileFilename,
        options.runtimeOptions->runtimeProfilingTypeMask,
        options.runtimeOptions->profilerDebuginfo);
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
