//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_INSTRUMENTS_H
#define SUPPORT_TELEMETRY_INSTRUMENTS_H

#include "Support/Telemetry/ForwardDecls.h"
#include <chrono>
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

  Counter(Counter &&) = default;
  Counter &operator=(Counter &&) = default;

private:
  friend class TelemetryContext;

  Counter(std::unique_ptr<opentelemetry::metrics::Counter<T>> counter)
      : counter(std::move(counter)) {}

  std::unique_ptr<opentelemetry::metrics::Counter<T>> counter;
};

#endif

// -------- Histogram and Timer --------

#ifndef MODULAR_ENABLE_TELEMETRY

template <typename T>
class Histogram {
public:
  void record(T value) {}

private:
  friend class TelemetryContext;

  Histogram() {}
};

template <typename T, typename DurationT = std::chrono::nanoseconds>
class Timer {
private:
  friend class TelemetryContext;
  /// TODO: Allow passing in attributes to attach to the recorded entry.
  Timer() {}
};

#else

template <typename T>
class Histogram {
public:
  void record(T value) { histogram->Record(value, context); }

  Histogram(Histogram &&) = default;
  Histogram &operator=(Histogram &&) = default;

private:
  friend class TelemetryContext;

  Histogram(std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram)
      : histogram(std::move(histogram)) {}

  std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
  opentelemetry::context::Context context{};
};

template <typename T, typename DurationT = std::chrono::nanoseconds>
class Timer {
  using ClockType = std::chrono::high_resolution_clock;
  using TimePointType = std::chrono::time_point<ClockType>;

public:
  ~Timer() {
    // The histogram pointer in the destructor may be null if the Timer was
    // moved.
    if (histogram) {
      auto end = ClockType::now();
      auto duration = std::chrono::duration_cast<DurationT>(end - start);
      histogram->Record(duration.count(), context);
    }
  }

  Timer(Timer &&) = default;
  Timer &operator=(Timer &&) = default;

private:
  friend class TelemetryContext;

  /// TODO: Allow passing in attributes to attach to the recorded entry.
  Timer(std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram)
      : histogram(std::move(histogram)) {
    start = ClockType::now();
  }

  std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
  opentelemetry::context::Context context{};
  /// The start time.
  TimePointType start;
};

#endif

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_INSTRUMENTS_H
