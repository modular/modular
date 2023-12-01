//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_INSTRUMENTS_H
#define SUPPORT_TELEMETRY_INSTRUMENTS_H

#include "Support/Telemetry/ForwardDecls.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <chrono>
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/metrics/sync_instruments.h"
#include "opentelemetry/sdk/resource/resource.h"
#endif // MODULAR_ENABLE_TELEMETRY

namespace M::Telemetry {

// -------- Counter --------

#ifndef MODULAR_ENABLE_TELEMETRY

using MetricAttributeValue =
    std::variant<bool, int32_t, int64_t, uint32_t, double, const char *,
                 std::string, uint64_t, llvm::StringRef>;

using AttributeMap = std::unordered_map<std::string, MetricAttributeValue>;

template <typename T>
class Counter {
public:
  void add(T value) {}

private:
  friend class TelemetryContext;

  Counter() {}
};

#else // MODULAR_ENABLE_TELEMETRY

using MetricAttributeValue = opentelemetry::common::AttributeValue;
// The "Record" methods in OTel are very particular with the typing --
// you need to use a KeyValueIterableView, which has a specific implementation
// that makes some assumptions about the iterator you pass in (namely that each
// entry has a `first` attribute you can access directly.) Needs to be an
// unordered_map because DenseMap seems not to work with owned strings.
using AttributeMap = std::unordered_map<std::string, MetricAttributeValue>;

template <typename T>
class Counter {
public:
  void add(T value) {
    auto attrs =
        opentelemetry::common::KeyValueIterableView<decltype(attributes)>{
            attributes};
    counter->Add(value, attributes);
  }

  // Allow overriding attributes and context.
  template <class... TArgs>
  void add(T value, TArgs &&...args) {
    counter->Add(value, std::forward<TArgs>(args)...);
  }

  Counter(Counter &&) = default;
  Counter &operator=(Counter &&) = default;

private:
  friend class TelemetryContext;

  Counter(std::unique_ptr<opentelemetry::metrics::Counter<T>> counter,
          const llvm::StringMap<MetricAttributeValue> &additionalAttributes)
      : counter(std::move(counter)) {
    for (auto &attr : additionalAttributes) {
      attributes[attr.first().str()] = attr.second;
    }
  }

  std::unique_ptr<opentelemetry::metrics::Counter<T>> counter;
  AttributeMap attributes;
};

#endif // MODULAR_ENABLE_TELEMETRY

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

template <typename T, typename DurationT>
class Timer {
private:
  friend class TelemetryContext;
  /// TODO: Allow passing in attributes to attach to the recorded entry.
  Timer() {}
};

#else // MODULAR_ENABLE_TELEMETRY

template <typename T>
class Histogram {
public:
  void record(T value) {
    auto attrs =
        opentelemetry::common::KeyValueIterableView<decltype(attributes)>{
            attributes};
    histogram->Record(value, attrs, context);
  }

  // Allow overriding attributes and context.
  template <class... TArgs>
  void record(T value, TArgs &&...args) {
    histogram->Record(value, std::forward<TArgs>(args)...);
  }

  Histogram(Histogram &&) = default;
  Histogram &operator=(Histogram &&) = default;
  AttributeMap attributes;

private:
  friend class TelemetryContext;

  Histogram(std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram,
            const llvm::StringMap<MetricAttributeValue> &additionalAttributes)
      : histogram(std::move(histogram)) {
    for (auto &attr : additionalAttributes) {
      attributes[attr.first().str()] = attr.second;
    }
  }

  std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
  opentelemetry::context::Context context{};
};

template <typename T, typename DurationT>
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
      auto attrs =
          opentelemetry::common::KeyValueIterableView<decltype(attributes)>{
              attributes};
      histogram->Record(duration.count(), attrs, context);
    }
  }

  Timer(Timer &&) = default;
  Timer &operator=(Timer &&) = default;

private:
  friend class TelemetryContext;

  Timer(std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram,
        const llvm::StringMap<MetricAttributeValue> &additionalAttributes)
      : histogram(std::move(histogram)) {
    for (auto &attr : additionalAttributes) {
      attributes[attr.first().str()] = attr.second;
    }
    start = ClockType::now();
  }

  std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
  opentelemetry::context::Context context{};
  AttributeMap attributes;
  /// The start time.
  TimePointType start;
};

#endif // MODULAR_ENABLE_TELEMETRY

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_INSTRUMENTS_H
