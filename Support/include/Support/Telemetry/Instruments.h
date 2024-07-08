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

using MetricAttributeValue = std::variant<bool, int32_t, uint32_t, int64_t,
                                          double, std::string, uint64_t>;

using MetricAttributeMap =
    std::unordered_map<std::string, MetricAttributeValue>;

#ifndef MODULAR_ENABLE_TELEMETRY

template <typename T>
class Counter {
public:
  void add(T value) {}

private:
  friend class TelemetryContext;

  Counter() {}
};

#else // MODULAR_ENABLE_TELEMETRY

template <typename T>
class Counter {
public:
  void
  add(T value,
      std::initializer_list<std::pair<llvm::StringRef, MetricAttributeValue>>
          additionalAttributes = {}) {
    std::unordered_map<std::string, opentelemetry::common::AttributeValue>
        attrs;
    for (auto &attr : owned_attributes) {
      std::visit([&](auto &v) { attrs[attr.first] = v; }, attr.second);
    }

    for (auto &attr : additionalAttributes) {
      std::visit([&](auto &v) { attrs[attr.first.str()] = v; }, attr.second);
    }

    counter->Add(value, attrs);
  }

  Counter(Counter &&) = default;
  Counter &operator=(Counter &&) = default;

private:
  friend class TelemetryContext;

  Counter(std::unique_ptr<opentelemetry::metrics::Counter<T>> counter,
          const llvm::StringMap<MetricAttributeValue> &additionalAttributes)
      : counter(std::move(counter)) {
    for (auto &attr : additionalAttributes) {
      owned_attributes[attr.first().str()] = attr.second;
    }
  }

  std::unique_ptr<opentelemetry::metrics::Counter<T>> counter;
  MetricAttributeMap owned_attributes;
};

#endif // MODULAR_ENABLE_TELEMETRY

// -------- Gauges ---------
#ifndef MODULAR_ENABLE_TELEMETRY

template <typename T>
class Gauge {
public:
  void add(T value) {}

private:
  friend class TelemetryContext;

  Gauge() {}
};

#else // MODULAR_ENABLE_TELEMETRY

template <typename T>
class Gauge {
public:
  void
  add(T value,
      std::initializer_list<std::pair<llvm::StringRef, MetricAttributeValue>>
          additionalAttributes = {}) {
    std::unordered_map<std::string, opentelemetry::common::AttributeValue>
        attrs;
    for (auto &attr : owned_attributes) {
      std::visit([&](auto &v) { attrs[attr.first] = v; }, attr.second);
    }

    for (auto &attr : additionalAttributes) {
      std::visit([&](auto &v) { attrs[attr.first.str()] = v; }, attr.second);
    }

    gauge->Add(value, attrs);
  }

  Gauge(Gauge &&) = default;
  Gauge &operator=(Gauge &&) = default;

private:
  friend class TelemetryContext;

  Gauge(std::unique_ptr<opentelemetry::metrics::UpDownCounter<T>> counter,
        const llvm::StringMap<MetricAttributeValue> &additionalAttributes)
      : gauge(std::move(counter)) {
    for (auto &attr : additionalAttributes) {
      owned_attributes[attr.first().str()] = attr.second;
    }
  }

  std::unique_ptr<opentelemetry::metrics::UpDownCounter<T>> gauge;
  MetricAttributeMap owned_attributes;
};

#endif // MODULAR_ENABLE_TELEMETRY

// -------- Histogram and Timer --------

#ifndef MODULAR_ENABLE_TELEMETRY

template <typename T>
class Histogram {
public:
  void
  record(T value,
         std::initializer_list<std::pair<llvm::StringRef, MetricAttributeValue>>
             additionalAttributes = {}) {}

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

public:
  void setAttribute(const std::string &key, const MetricAttributeValue &value) {
  }
};

#else // MODULAR_ENABLE_TELEMETRY

template <typename T>
class Histogram {
public:
  void
  record(T value,
         std::initializer_list<std::pair<llvm::StringRef, MetricAttributeValue>>
             additionalAttributes = {}) {
    std::unordered_map<std::string, opentelemetry::common::AttributeValue>
        attrs;
    for (auto &attr : owned_attributes) {
      std::visit([&](auto &v) { attrs[attr.first] = v; }, attr.second);
    }
    histogram->Record(value, attrs, context);
  }

  Histogram(Histogram &&) = default;
  Histogram &operator=(Histogram &&) = default;

private:
  friend class TelemetryContext;

  Histogram(std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram,
            const llvm::StringMap<MetricAttributeValue> &additionalAttributes)
      : histogram(std::move(histogram)) {
    for (auto &attr : additionalAttributes) {
      owned_attributes[attr.first().str()] = attr.second;
    }
  }

  std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
  opentelemetry::context::Context context{};
  MetricAttributeMap owned_attributes;
};

template <typename T, typename DurationT>
class Timer {
  using ClockType = std::chrono::high_resolution_clock;
  using TimePointType = std::chrono::time_point<ClockType>;

public:
  void setAttribute(const std::string &key, const MetricAttributeValue &value) {
    owned_attributes[key] = value;
  }

  ~Timer() {
    // The histogram pointer in the destructor may be null if the Timer was
    // moved.
    if (histogram) {
      auto end = ClockType::now();
      auto duration = std::chrono::duration_cast<DurationT>(end - start);
      std::unordered_map<std::string, opentelemetry::common::AttributeValue>
          attrs;
      for (auto &attr : owned_attributes) {
        std::visit([&](auto &v) { attrs[attr.first] = v; }, attr.second);
      }
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
      owned_attributes[attr.first().str()] = attr.second;
    }

    start = ClockType::now();
  }

  std::unique_ptr<opentelemetry::metrics::Histogram<T>> histogram;
  opentelemetry::context::Context context{};

  // We need to take ownership of these attributes, and the expected
  // `AttributeValue` does not work for this since those are all references.
  std::unordered_map<std::string, MetricAttributeValue> owned_attributes;
  /// The start time.
  TimePointType start;
};

#endif // MODULAR_ENABLE_TELEMETRY

} // namespace M::Telemetry

#endif // SUPPORT_TELEMETRY_INSTRUMENTS_H
