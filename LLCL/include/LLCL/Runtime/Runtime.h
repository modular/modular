//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares top level "god object" that organizes LLCL thread pool,
// memory allocator, etc.
//
// This header file is intended to be a low-dependency header that other things
// compose on top of.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_H
#define LLCL_RUNTIME_H

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/AsyncValueRef.h"
#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/Chain.h"
#include "Support/Telemetry/Telemetry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>

namespace M {
class Error;
}

namespace M::LLCL {
class Allocator;
class WorkQueue;

/// This represents one instance of the LLCL runtime, which can have multiple
/// threads, a private heap for data, a way of reporting errors, and other
/// global context. This is also the natural unit for task cancellation.
class Runtime final {
public:
  /// Construct runtime with allocator and workQueue. If profileFilename is
  /// non-empty then time profiling will be activated and the profile JSON
  /// will be written to that file.
  Runtime(std::unique_ptr<Allocator> allocator,
          std::unique_ptr<WorkQueue> workQueue, StringRef profileFilename = {});
  ~Runtime();

  /// Return a CompactRuntimePtr that identifies this Runtime instance.
  CompactRuntimePtr getCompactPtr() const {
    return CompactRuntimePtr(runtimeIndex);
  }

  /// Return a reference to a pre-allocated Chain value that is already ready.
  /// This can be used by logic that needs to flag that a side effect has
  /// already happened, without doing an extraneous memory allocation.
  const AsyncValueRef<Chain> &getReadyChain() const { return readyChain; }

  //===--------------------------------------------------------------------===//
  // Profiling and Telemetry
  //===--------------------------------------------------------------------===//

  /// Return a reference to the profiler instance, if its been initialized.
  std::optional<TimeTraceProfiler> &getProfiler() { return profiler; }

  /// Return a reference to this runtime's TelemetryContext instance. If
  /// telemetry is disabled, this will return NOOP instruments.
  M::Telemetry::TelemetryContext &getTelemetryContext() {
    return telemetryContext;
  }

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level allocator.
  Allocator *getAllocator() { return allocator.get(); }

  //===--------------------------------------------------------------------===//
  // Concurrency
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level WorkQueue.  You should typically
  /// interface with the higher level algorithms in Algorithms.h.
  WorkQueue *getWorkQueue() { return workQueue.get(); }

  //===--------------------------------------------------------------------===//
  // Cancel the current execution
  //===--------------------------------------------------------------------===//

  /// Cancel the current MEF Execution. This transitions this Runtime to the
  /// canceled state, which causes all asynchronously executing threads to be
  /// canceled when they check the cancellation state (e.g. in MEFExecutor).
  void cancelExecution(EncodedDiagnostic message);

  /// restartFromCancellation() transitions Runtime from the canceled state to
  /// the normal execution state.
  void restartFromCancellation();

  /// When this Runtime is in a canceled state, getCancelValue() returns a
  /// non-null AsyncValue containing the message for the cancellation.
  /// Otherwise, it returns nullptr.
  AsyncValue *getCancelValue() const {
    return cancelValue.load(std::memory_order_acquire);
  }

  //===--------------------------------------------------------------------===//
  // Configuration
  //===--------------------------------------------------------------------===//

  /// Emplaces a new configuration of type T into the runtime's set of
  /// configurations and returns a reference to it. The runtime can hold at
  /// most one configuration per T. The returned reference is stable for the
  /// life of the runtime. This method is not thread safe, and it is expected
  /// all required configurations are emplaced before any tasks using getConfig
  /// below begin execution.
  template <typename T, typename... Args>
  T &emplaceConfig(Args &&...args) {
    auto typeId = TypeID::get<T>();
    assert(!configs.contains(typeId.getDenseIndex()) &&
           "Runtime already holds configuration of type");
    auto ptr = std::make_unique<Config<T>>(std::forward<Args>(args)...);
    T &result = ptr->payload;
    configs.insert({typeId.getDenseIndex(), std::move(ptr)});
    return result;
  }

  /// Returns a pointer to the configuration of type T held by the runtime,
  /// or nullptr if no such configuration is held. This method is thread safe,
  /// on the assumption no further calls to emplaceConfig will be made.
  template <typename T>
  const T *getConfig() const {
    auto typeId = TypeID::get<T>();
    auto itr = configs.find(typeId.getDenseIndex());
    if (itr == configs.end())
      return nullptr;
    auto ptr = static_cast<Config<T> *>(itr->second.get());
    return &ptr->payload;
  }

private:
  Runtime(const Runtime &) = delete;
  void operator=(const Runtime &) = delete;

  /// The 'signature' for the type id registration system the runtime depends
  /// on. This is expected to be unique for the running process. This can be
  /// used to catch, at runtime, accidental multiple definitions for Modular
  /// runtime statics across dynamic libraries / executables.
  intptr_t signature;

  /// These are the allocator and workQueue's that were configured by the client
  /// for this Runtime.
  std::unique_ptr<Allocator> allocator;
  std::unique_ptr<WorkQueue> workQueue;

  /// Filename into which time profiling should be written, or the empty
  /// string if disabled.
  std::string profileFilename;

  /// An active profiler used for the runtime, or nullopt if profiling is
  /// disabled. This is only set when profileFilename is non-empty.
  std::optional<TimeTraceProfiler> profiler;

  /// The TelemetryContext instance used by the runtime. It returns NOOP
  /// instruments if telemetry is disabled.
  M::Telemetry::TelemetryContext telemetryContext;

  /// This is the index # for the runtime object created.  This is held by the
  /// CompactRuntimePtr.
  uint8_t runtimeIndex;

  /// This is a preallocated Chain value that is marked as ready, for use by
  /// getReadyChain.
  AsyncValueRef<Chain> readyChain;

  /// If execution is cancelled, this holds the error value to forward into the
  /// results of computations.
  std::atomic<AsyncValue *> cancelValue{nullptr};

  /// Base class for configuration objects.
  struct ConfigBase {
    ConfigBase() = default;

    /// Though we have TypeID::getValueDestructor, and could with a bit of
    /// hackery destroy any Config<T> given only TypeID::get<T>(), it requires
    /// ensuring Config::payload is always placed after this object. I think
    /// that's too obscure to be worth avoiding the vtable, and adds a ton
    /// of padding anyways.
    virtual ~ConfigBase() = default;
  };

  /// Holds the payload for configuration object of type T.
  template <typename T>
  struct Config : public ConfigBase {
    T payload;
    template <typename... Args>
    Config(Args &&...args) : payload(std::forward<Args>(args)...) {}
    ~Config() override = default;
  };

  /// A map from globally unique type identifiers TypeID::get<T>() (using
  /// their 'dense index' form) to an instance of Config<T>.
  DenseMap<size_t, std::unique_ptr<ConfigBase>> configs;

  friend void checkUniqueRuntime(const Runtime &runtime);
};

/// In debug builds, assert the given runtime's 'signature' agrees with what
/// the host's idea of signature for its dynamic library / executable.
/// This can be used to catch, at runtime, accidental multiple definitions for
/// Modular runtime statics across dynamic libraries / executables.
inline void checkUniqueRuntime(const Runtime &runtime) {
  assert(runtime.signature ==
             (TypeID::getSignature() ^ CompactRuntimePtr::getSignature()) &&
         "It appears your process has statically linked the Modular Runtime "
         "multiple times across dynamic library / executable boundaries. "
         "Please don't do that.");
}

} // namespace M::LLCL

#endif // LLCL_RUNTIME_H
