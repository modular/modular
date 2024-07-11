//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a basic set of command line options for setting up and
// configuring an M::AsyncRT::Runtime for tools to use.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_RUNTIMECLOPTIONS_H
#define ASYNCRT_RUNTIME_RUNTIMECLOPTIONS_H

#include "AsyncRT/Runtime/Runtime.h"
#include "Support/ADT/GenericUniquePtrSet.h"
#include "Support/CommandLine.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/RCRef.h"
#include "llvm/Support/Threading.h"
#include <chrono>
#include <thread>
#include <type_traits>

namespace M::AsyncRT {

class Runtime;

/// Contains a number of command-line options that are shared among binaries
/// that use the LLCL Runtime and want configurability of Allocator,
/// WorkQueue, stopping behavior, etc.
///
class RuntimeCLOptions {
public:
  RuntimeOptions &options;
  RuntimeCLOptions(RuntimeOptions &o) : options(o) {}

private:
  llvm::cl::OptionCategory RuntimeOptionsCategory{
      "Runtime command line options"};
  //===--------------------------------------------------------------------===//
  // Core Runtime configuration.
  //===--------------------------------------------------------------------===//
  // Enable HostAllocator types to be specified on the command line.
  M::cl::MOpt<RuntimeOptions::WorkQueueType, true> workQueueType{
      "workqueue", llvm::cl::desc("Specify workqueue type:"),
      llvm::cl::values(
          clEnumValN(RuntimeOptions::WorkQueueType::kDefault, "default",
                     "Auto-select based on # threads"),
          clEnumValN(RuntimeOptions::WorkQueueType::kSingleThread,
                     "single-thread",
                     "Work queue that only ever uses one thread"),
          clEnumValN(RuntimeOptions::WorkQueueType::kThreadPool, "thread-pool",
                     "Default threaded work queue based on std::thread")),
      llvm::cl::location(options.workQueueType),
      llvm::cl::cat(RuntimeOptionsCategory)};

  // Enable HostAllocator types to be specified on the command line.
  M::cl::MOpt<RuntimeOptions::AllocatorType, true> allocatorType{
      "allocator", llvm::cl::desc("Specify allocator type:"),
      llvm::cl::values(

          clEnumValN(RuntimeOptions::AllocatorType::kMalloc, "malloc",
                     "System malloc/free"),
          clEnumValN(RuntimeOptions::AllocatorType::kTCMalloc, "tcmalloc",
                     "TCMalloc new/delete. Not available on all targets"),
          clEnumValN(RuntimeOptions::AllocatorType::kLeakChecker,
                     "leak-checker", "Allocator with leak checking"),
          clEnumValN(RuntimeOptions::AllocatorType::kProfiler, "profiler",
                     "Allocator with profiling and leak checking"),
          clEnumValN(RuntimeOptions::AllocatorType::kUseAfterFree,
                     "use-after-free",
                     "Allocator to detect use-after-free errors. Not available "
                     "on all targets.")),
      llvm::cl::location(options.allocatorType),
      llvm::cl::cat(RuntimeOptionsCategory)};

  // Specify the number of threads. If `thread==1`, then we automatically set
  // our work queue to `WorkQueueType::kSingleThread`. Otherwise, we assume
  // the work queue is using a thread pool. The default number of threads is
  // the result of M::getNumThreads().
  M::cl::MOpt<size_t, true> numThreads{
      "num-threads",
      llvm::cl::desc("Specify the number of threads to run the work queue "
                     "items. If zero "
                     "(default), will be chosen by heuristics."),
      llvm::cl::location(options.numThreads),
      llvm::cl::cat(RuntimeOptionsCategory)};
  M::cl::MOpt<size_t, true> maxThreads{
      "max-threads",
      llvm::cl::desc("Bound num-threads in the case of auto-configuration."),
      llvm::cl::location(options.maxThreads),
      llvm::cl::cat(RuntimeOptionsCategory)};

  // Specify the amount of time a worker thread should spin for before
  // sleeping. The optimal value here depends on the system latency for thread
  // sleep and wake-up, as well as other external factors like how many other
  // threadpools are sharing the system.
  M::cl::MOpt<size_t, true> threadBusyWaitTime{
      "thread-busy-wait-time-us",
      llvm::cl::desc(
          "Specify the number of microseconds for threads to spin before "
          "locking. Zero indicates that threads should never spin."),
      llvm::cl::location(options.threadBusyWaitTime),
      llvm::cl::cat(RuntimeOptionsCategory)};

  // Specify whether the workqueue should be created using thread affinity.
  M::cl::MOpt<bool, true> cpuAffinity{
      "cpu-affinity",
      llvm::cl::desc("Assign CPU affinity to threads within the work queue."),
      llvm::cl::location(options.withAffinity),
      llvm::cl::cat(RuntimeOptionsCategory)};

#if MODULAR_PARANOID
  /// If true, and in a MODULAR_PARANOID build, perform additional (and
  /// very expensive!) runtime actions to make race conditions and other
  /// undefined behaviour more likely to be observed by unit tests.
  M::cl::MOpt<bool, true> paranoid{"paranoid",
                                   llvm::cl::desc("Turn on paranoid mode"),
                                   llvm::cl::location(options.paranoid),
                                   llvm::cl::cat(RuntimeOptionsCategory)};
#endif

  // Filename to hold the time profiling output (as JSON text).
  M::cl::MOpt<std::string, true> profileFilename{
      "time-profile",
      llvm::cl::desc(
          kIsProfilingEnabled
              ? "Specify the filename base for profiling output. The tracing "
                "data will be written to a file called "
                "\"<base>.time-trace\". "
                "This will be a JSON text in the standard profiling format. "
                "The events will be written to a text file called "
                "\"<base>.time-events.csv\". An empty filename base disables "
                "profiling (the default)."
              : "Specify the filename base for profiling output. WARNING: "
                "This "
                "option is ignored in this build. Rebuild with "
                "MODULAR_LLCL_MAX_PROFILING_LEVEL greater than 0 to enable "
                "it."),
      llvm::cl::location(options.profileFilename),
      llvm::cl::cat(RuntimeOptionsCategory)};

  // Should we generate debuginfo for a profiler?
  M::cl::MOpt<RuntimeOptions::ProfilerDebuginfo, true> profilerDebuginfo{
      "profiler-debuginfo",
      llvm::cl::desc(
          "Output debug symbols in a way that a profiler can understand. After "
          "running under perf, use `perf inject --jit -i perf.data -o "
          "perf.jit.data` to add debug info for kernels to the profile."),
      llvm::cl::values(
          clEnumValN(RuntimeOptions::ProfilerDebuginfo::kNoProfiler, "none",
                     "Do not generate debuginfo"),
          clEnumValN(RuntimeOptions::ProfilerDebuginfo::kPerfProfiler, "perf",
                     "Generate debuginfo for perf."),
          clEnumValN(RuntimeOptions::ProfilerDebuginfo::kSOProfiler, "so",
                     "Generate debuginfo by loading compiled kernels into a "
                     "shared library. Should work with all profilers.")),
      llvm::cl::location(options.profilerDebuginfo),
      llvm::cl::cat(RuntimeOptionsCategory)};

  /// Set the behavior of executors if one of the functions they should run
  /// returns with an error. E.g. Set to `continue` for diagnostic
  /// verification.
  M::cl::MOpt<RuntimeOptions::OnFailure, true> onFailure{
      "on-failure",
      llvm::cl::desc("Behavior in case an executed function returns with an "
                     "error. Ignored if there is only one function executed."),
      llvm::cl::values(clEnumValN(RuntimeOptions::OnFailure::kContinue,
                                  "continue", "System malloc/free"),
                       clEnumValN(RuntimeOptions::OnFailure::kExit, "exit",
                                  "Allocator with leak checking")),
      llvm::cl::location(options.onFailure)};
};

} // namespace M::AsyncRT

#endif // ASYNCRT_RUNTIME_RUNTIMECLOPTIONS_H
