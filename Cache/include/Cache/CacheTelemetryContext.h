//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHE_TELEMETRY_CONTEXT_H
#define CACHE_CACHE_TELEMETRY_CONTEXT_H

#include "Support/Context.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/OpDefinition.h"

namespace M::AsyncRT {
class Runtime;
}

namespace M::Cache {
/// Utility that manages the objects used to perform telemetry related to the
/// KGEN compiler.
class CacheTelemetryContext {
public:
  CacheTelemetryContext(Telemetry::TelemetryContext &ctx);

  static CacheTelemetryContext &getCacheTelemetryContext(ContextRef context);
  static CacheTelemetryContext &getCacheTelemetryContext(Context *context);

  /// Record a cache hit event.
  void recordCacheHit(llvm::StringRef pipelineName);

  /// Record a cache miss event.
  void recordCacheMiss(llvm::StringRef pipelineName);

  static std::function<void(mlir::Operation *)> getTelemetryOnMissLambda(
      const std::string &counterName, const std::string &timerName,
      const llvm::StringMap<M::Telemetry::MetricAttributeValue> &attributes =
          {});

  static std::function<void(mlir::Operation *)>
  getTelemetryOnHitLambda(const std::string &counterName);

private:
  Telemetry::Counter<uint64_t> cacheHitCounter;
  Telemetry::Counter<uint64_t> cacheMissCounter;
};

} // namespace M::Cache

#endif // CACHE_CACHE_TELEMETRY_CONTEXT_H
