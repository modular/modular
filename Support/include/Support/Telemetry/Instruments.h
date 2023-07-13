//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_INSTRUMENTS_H
#define SUPPORT_TELEMETRY_INSTRUMENTS_H

#include "Support/Telemetry/ForwardDecls.h"
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/metrics/sync_instruments.h"
#endif

namespace M::Telemetry {

// -------- Counter --------

#ifndef MODULAR_ENABLE_TELEMETRY

template <typename T>
class Counter {
public:
  void add(T value) {}

private:
  friend class TelemetryContext;

  Counter() {}
};

#else

template <typename T>
class Counter {
public:
  void add(T value) { counter->Add(value); }

private:
  friend class TelemetryContext;

  Counter(std::unique_ptr<opentelemetry::metrics::Counter<T>> counter)
      : counter(std::move(counter)) {}

  std::unique_ptr<opentelemetry::metrics::Counter<T>> counter;
};

#endif

// -------- Histogram --------

#ifndef MODULAR_ENABLE_TELEMETRY

template <typename T>
class Histogram {
public:
  void record(T value) {}

private:
  friend class TelemetryContext;

  Histogram() {}
};

#else

template <typename T>
class Histogram {
public:
  void record(T value) { histogram->Record(value); }

private:
  friend class TelemetryContext;

  Histogram(std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram)
      : histogram(std::move(histogram)) {}

  std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
};

#endif

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_INSTRUMENTS_H
