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
#include <filesystem>
#include <mutex>
#include <sstream>
#endif // MODULAR_ENABLE_TELEMETRY

namespace M::LLCL::Telemetry {

class ManualExportingMetricReader;

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
  /// parameter.
  template <typename DurationT = std::chrono::nanoseconds>
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

  /// Flush all the collected metrics.
  void flush();

  /// This struct provides an RAII-style way to flush telemetry at the end of a
  /// scope.
  struct AutoFlush {
    AutoFlush(LLCL::RCRef<TelemetryContext> ctx) : context(std::move(ctx)) {}
    ~AutoFlush() { context->flush(); }

    LLCL::RCRef<TelemetryContext> context;
  };

  /// Get an AutoFlush object from `this`.
  AutoFlush autoFlush() {
    return AutoFlush(LLCL::RCRef<TelemetryContext>::copy(this));
  }

private:
#ifdef MODULAR_ENABLE_TELEMETRY
  // Metrics.
  std::unique_ptr<opentelemetry::metrics::MeterProvider> metricsProvider;
  std::shared_ptr<opentelemetry::metrics::Meter> meter;
  std::shared_ptr<ManualExportingMetricReader> metricReader;
  // Logs.
  std::shared_ptr<opentelemetry::logs::LoggerProvider> loggerProvider;
  std::shared_ptr<opentelemetry::logs::EventLoggerProvider> eventLoggerProvider;

  /// We must not call export concurrently for the same exporter instance, so we
  /// need to make sure we lock it.
  std::mutex exportLock;
#endif
};

} // namespace M::LLCL::Telemetry

#endif // SUPPORT_TELEMETRY_H
