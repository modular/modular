//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_H
#define SUPPORT_TELEMETRY_H

#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "LLCL/Support/Telemetry/Instruments.h"
#include "LLCL/Support/Telemetry/Logs.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <variant>
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/logs/event_logger_provider.h"
#include "opentelemetry/logs/logger_provider.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/meter_provider.h"
#endif // MODULAR_ENABLE_TELEMETRY

namespace M::LLCL::Telemetry {

// TODO: Support some of these in config file.
/// When the TelemetryContext is destroyed, it does a synchronous flush to
/// ensure that any telemetry that hasn't yet been exported is exported. This
/// timeout is how long it waits for the export to complete before the
/// destructor returns.
constexpr auto kShutdownFlushTimeout = std::chrono::milliseconds(100);
/// Periodically export metrics every kExportInterval duration.
constexpr auto kExportInterval = std::chrono::milliseconds(10000);
/// Timeout for periodic metric exports. Note that periodic exports happen
/// asynchronously and this timeout is for the worker thread that does them
/// (OTel-managed thread). NOTE: this value must be smaller than the export
/// interval.
constexpr auto kExportTimeout = std::chrono::milliseconds(1000);

// TODO: Add ways to organize instruments (e.g. Meters/instrumentation scope)
// later if needed.

/// A TelemetryContext provides access to instruments (e.g. Counter, Histogram)
/// to instrument the code and generate metrics. These metrics will be exported
/// by the TelemetryContext based on the options passed to it during creation.
///
/// Right now we are assuming that the TelemetryContext will collect Resource
/// attributes (e.g. CPU info, OS info, version of software components) without
/// this information being passed to it explicitly through its API, but this is
/// subject to change.
class TelemetryContext : public LLCL::ReferenceCounted<TelemetryContext> {
public:
  /// This is just a copy of the OTel OwnedAttributeValue - we can use this to
  /// provide resources to the telemetry context. We don't support the lists
  /// yet, we can add those as necessary.
  using AttributeValue =
      std::variant<bool, int32_t, int64_t, uint32_t, double, StringRef,
                   ArrayRef<bool>, ArrayRef<int32_t>, ArrayRef<int64_t>,
                   ArrayRef<uint32_t>, ArrayRef<double>, uint64_t,
                   ArrayRef<uint64_t>, ArrayRef<uint8_t>>;

  /// Construct a TelemetryContext with additional resource strings. These will
  /// be added to the OTel resources that are attached to every log message.
  TelemetryContext(const llvm::StringMap<AttributeValue> &resources = {});

  ~TelemetryContext();

  // XXX: not sure if it's better to allocate Counter and Histogram on the heap
  // or not. For Otel, the Counter struct will basically just contain a pointer
  // to the Otel counter, and so returning the struct seems appropiate.

  /// Create a Counter<uint64_t>.
  Counter<uint64_t> createUInt64Counter(StringRef name,
                                        StringRef description = "",
                                        StringRef unit = "") {
    // TODO: If the name is invalid, it looks like OTel logs the error and
    // returns a NOOP counter. Instead, we should probably try to assert that
    // the name is valid or that the returned counter is not NOOP. Same for
    // other instruments.
#ifdef MODULAR_ENABLE_TELEMETRY
    return Counter<uint64_t>(meter->CreateUInt64Counter(
        name.data(), description.data(), unit.data()));
#else
    return Counter<uint64_t>();
#endif
  }

  /// Create a Counter<double>.
  Counter<double> createDoubleCounter(StringRef name,
                                      StringRef description = "",
                                      StringRef unit = "") {
#ifdef MODULAR_ENABLE_TELEMETRY
    return Counter<double>(meter->CreateDoubleCounter(
        name.data(), description.data(), unit.data()));
#else
    return Counter<double>();
#endif
  }

  /// Create a Histogram<uint64_t>.
  Histogram<uint64_t> createUInt64Histogram(StringRef name,
                                            StringRef description = "",
                                            StringRef unit = "") {
#ifdef MODULAR_ENABLE_TELEMETRY
    return Histogram<uint64_t>(meter->CreateUInt64Histogram(
        name.data(), description.data(), unit.data()));
#else
    return Histogram<uint64_t>();
#endif
  }

  /// Create a Histogram<double>.
  Histogram<double> createDoubleHistogram(StringRef name,
                                          StringRef description = "",
                                          StringRef unit = "") {
#ifdef MODULAR_ENABLE_TELEMETRY
    return Histogram<double>(
        meter->CreateDoubleHistogram(name, description, unit));
#else
    return Histogram<double>();
#endif
  }

  /// Create a Timer. If unit is omitted, the method will implicitly set
  /// it to one of {"ns", "us", "ms", "s"} based on the DurationT template
  /// parameter (e.g. std::chrono::microseconds).
  template <typename DurationT>
  Timer<uint64_t, DurationT> createUInt64Timer(StringRef name,
                                               StringRef description = "",
                                               StringRef unit = "") {
#ifdef MODULAR_ENABLE_TELEMETRY
    if (unit.empty()) {
      if constexpr (std::is_same_v<DurationT, std::chrono::nanoseconds>)
        unit = "ns";
      else if constexpr (std::is_same_v<DurationT, std::chrono::microseconds>)
        unit = "us";
      else if constexpr (std::is_same_v<DurationT, std::chrono::milliseconds>)
        unit = "ms";
      else if constexpr (std::is_same_v<DurationT, std::chrono::seconds>)
        unit = "s";
    }
    return Timer<uint64_t, DurationT>(
        meter->CreateUInt64Histogram(name, description, unit));
#else
    return Timer<uint64_t, DurationT>();
#endif
  }

  /// Create a Logger with given domain (see
  /// https://opentelemetry.io/docs/specs/otel/logs/semantic_conventions/events/).
  std::shared_ptr<Logs::Logger> getLogger(StringRef eventDomain) {
#ifdef MODULAR_ENABLE_TELEMETRY
    auto otelLogger = loggerProvider->GetLogger("modular_logger");
    auto otelEventLogger =
        eventLoggerProvider->CreateEventLogger(otelLogger, eventDomain);
    return std::shared_ptr<Logs::Logger>(new Logs::Logger(otelEventLogger));
#else
    return std::shared_ptr<Logs::Logger>(new Logs::Logger());
#endif
  }

  /// Flush all the collected metrics. Blocks until the flush completes
  /// or the timeout elapses, whichever comes first.
  /// NOTE: TelemetryContext flushes periodically asynchronously. Manual
  /// flushing is not recommended except where needed (for example the
  /// TelemetryContext flushes itself at shutdown).
  void
  flush(std::chrono::microseconds timeout = std::chrono::microseconds::max());

  /// This struct provides an RAII-style way to flush telemetry at the end of a
  /// scope.
  struct AutoFlush {
    AutoFlush(LLCL::RCRef<TelemetryContext> ctx,
              std::chrono::microseconds timeout)
        : context(std::move(ctx)), timeout(timeout) {}
    ~AutoFlush() { context->flush(timeout); }

    LLCL::RCRef<TelemetryContext> context;
    std::chrono::microseconds timeout;
  };

  /// Get an AutoFlush object from `this`. The object will flush when it goes
  /// out of scope, blocking until the flush completes or the timeout elapses,
  /// whichever comes first. NOTE: TelemetryContext flushes periodically
  /// asynchronously. Flushing with scoped autoflush is not generally
  /// recommended.
  AutoFlush autoFlush(
      std::chrono::microseconds timeout = std::chrono::microseconds::max()) {
    return AutoFlush(LLCL::RCRef<TelemetryContext>::copy(this), timeout);
  }

private:
#ifdef MODULAR_ENABLE_TELEMETRY
  // Metrics.
  std::unique_ptr<opentelemetry::metrics::MeterProvider> metricsProvider;
  std::shared_ptr<opentelemetry::metrics::Meter> meter;
  //  Logs.
  std::shared_ptr<opentelemetry::logs::LoggerProvider> loggerProvider;
  std::shared_ptr<opentelemetry::logs::EventLoggerProvider> eventLoggerProvider;
#endif
};

} // namespace M::LLCL::Telemetry

#endif // SUPPORT_TELEMETRY_H
