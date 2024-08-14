//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Init/Init.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Config/Version.h"
#include "Support/Configuration.h"
#include "Support/Context.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Settings/Settings.h"
#include "Support/Telemetry/Telemetry.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;

static void logHostMachineInfo(llvm::raw_fd_ostream &crashLog) {
  auto hostMachineOr = M::getHostMachineInfo();
  if (hostMachineOr.isError()) {
    crashLog
        << "Failed to get machine info.  Not including it in the crash log."
        << '\n';
  } else {
    crashLog << "Host machine info below:" << '\n';
    M::HostMachineInfo hostInfo = hostMachineOr.takeValue();
    hostInfo.print(crashLog);
  }
}

static void crashHandler(void *context) {
  std::string *programName = reinterpret_cast<std::string *>(context);

  // Crash dumps saved by Crashpad first and then the logging here is reached
  // after Crashpad re-raises to our previously registered signal handler.
  llvm::errs() << *programName << " crashed!\n";
  llvm::errs() << "Please file a bug report.\n";

  // As a useful helper, always print a full name when there is an environment
  // variable named "CI". Perhaps this is also useful to end users and should
  // be documented?
  if (std::getenv("CI")) {
    llvm::sys::PrintStackTrace(llvm::errs());
    llvm::errs() << '\n';
    logHostMachineInfo(llvm::errs());
  }
}

static void registerSignalHandler(StringRef programName) {
  // Ensure that the handler is only registered once.
  static std::string programNameStorage;
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() {
    programNameStorage = std::string(programName);
    llvm::sys::AddSignalHandler(crashHandler,
                                reinterpret_cast<void *>(&programNameStorage));
  });
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

  // Create the settings object. This will refresh the underlying entitlement
  // store if required by the provided policy.
  auto settingsOr = Settings::open(httpCtx.copy(), options.entitlementPolicy,
                                   options.refreshPolicy);
  if (settingsOr.isError())
    return settingsOr.takeError();
  Settings settings = std::move(*settingsOr);

  // If we have a certificate authority, set that on the HTTPContext.
  StringRef caInfo = settings.get<StringRef>("ssl.cainfo");
  if (!caInfo.empty())
    httpCtx->setCAInfo(std::string(caInfo));

  // Setup authentication on the HTTP client.
  httpCtx->setupAuth(settings.clientKeyPriv(), settings.clientCert());

  // Enable crash logging, if appropriate.
  if (!options.forceDisableCrashReporting &&
      settings.getBool("crash_reporting.enabled", true)) {
    initCrashpadForProgram(programName, &settings);
    registerSignalHandler(programName);
  }

  // Move everything into the context. Construct here may used the settings.
  ctx->emplace<HTTPContextRef>(std::move(httpCtx));
  ctx->emplace<TelemetryContext>(settings, options.resources);

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
                  options.runtimeOptions->poolName,
                  options.runtimeOptions->paranoid);
    ctx->emplace<Runtime>(runtimePtr, ctx.getPointer(), std::move(allocator),
                          std::move(workQueue),
                          profileFilename.empty()
                              ? options.runtimeOptions->profileFilename
                              : profileFilename,
                          options.runtimeOptions->runtimeProfilingTypeMask,
                          options.runtimeOptions->profilerDebuginfo);
  }

  // Finally move the settings.
  ctx->emplace<Settings>(std::move(settings));

  // Return the useable context.
  return std::move(ctx);
}
