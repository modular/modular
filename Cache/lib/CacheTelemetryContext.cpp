//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheTelemetryContext.h"
#include "AsyncRT/CompilerSupport/Context.h"

using namespace M;
using namespace M::Cache;

CacheTelemetryContext::CacheTelemetryContext(Telemetry::TelemetryContext &ctx)
    : cacheHitCounter(ctx.createUInt64Counter(
          "mojo.compile.cache.hit", Telemetry::Level::L2,
          /*attributes=*/{}, "Number of compilation cache hits.")),
      cacheMissCounter(ctx.createUInt64Counter(
          "mojo.compile.cache.miss", Telemetry::Level::L2,
          /*attributes=*/{}, "Number of compilation cache misses.")) {}

CacheTelemetryContext &
CacheTelemetryContext::getCacheTelemetryContext(Context *context) {
  auto &telemetryCtx = *context->get<M::Telemetry::TelemetryContext>();
  return context->emplaceIfMissing<CacheTelemetryContext>(telemetryCtx);
}

CacheTelemetryContext &
CacheTelemetryContext::getCacheTelemetryContext(ContextRef context) {
  return getCacheTelemetryContext(context.getPointer());
}

void CacheTelemetryContext::recordCacheHit(llvm::StringRef pipelineName) {
#ifdef MODULAR_ENABLE_TELEMETRY
  cacheHitCounter.add(1, {{"pipeline", pipelineName.str()}});
#endif // MODULAR_ENABLE_TELEMETRY
}

void CacheTelemetryContext::recordCacheMiss(llvm::StringRef pipelineName) {
#ifdef MODULAR_ENABLE_TELEMETRY
  cacheMissCounter.add(1, {{"pipeline", pipelineName.str()}});
#endif // MODULAR_ENABLE_TELEMETRY
}

std::function<void(mlir::Operation *)>
CacheTelemetryContext::getTelemetryOnMissLambda(
    const std::string &counterName, const std::string &timerName,
    const llvm::StringMap<M::Telemetry::MetricAttributeValue> &attrs) {
  return [counterName, timerName, attrs](mlir::Operation *op) {
#ifdef MODULAR_ENABLE_TELEMETRY
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheMiss(counterName);

    [[maybe_unused]] auto timeScope =
        loadContext(op->getContext())
            ->get<M::Telemetry::TelemetryContext>()
            ->createUInt64Timer<std::chrono::milliseconds>(
                timerName, M::Telemetry::Level::L2, attrs);
#endif
  };
}

std::function<void(mlir::Operation *)>
CacheTelemetryContext::getTelemetryOnHitLambda(const std::string &counterName) {
  return [counterName](mlir::Operation *op) {
#ifdef MODULAR_ENABLE_TELEMETRY
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheHit(counterName);

#endif
  };
}
