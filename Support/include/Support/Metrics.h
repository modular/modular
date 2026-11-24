//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// In-process metrics collection for the Modular runtime.
//
// Callers register named instruments once at startup via MetricsCollector and
// retain a direct reference for the lifetime of the process:
//
//   Counter& reqs = M::Metrics::metrics().registerCounter("my.requests");
//   reqs.increment();
//
// A periodic Python task calls collect() to snapshot all values and forward
// them to the OTel/Prometheus export pipeline. Counter and Gauge values are
// cumulative (the Python layer computes deltas); Histogram snapshots reset on
// each collect() call.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_METRICS_H
#define SUPPORT_METRICS_H

#include "Support/SymbolExport.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace M::Metrics {

// Counters hold a single monotonically-increasing value of type uint64_t.
class Counter {
  std::atomic<uint64_t> count = 0;

public:
  void increment() noexcept { count.fetch_add(1, std::memory_order_relaxed); }

  uint64_t read() const noexcept {
    return count.load(std::memory_order_relaxed);
  }
};

// Gauges are a single int64_t value which can change both up and down,
// as well as discontinuously.
class Gauge {
  std::atomic<int64_t> gauge = 0;

public:
  void decrease(int64_t delta = 1) noexcept {
    gauge.fetch_sub(delta, std::memory_order_relaxed);
  }

  void increase(int64_t delta = 1) noexcept {
    gauge.fetch_add(delta, std::memory_order_relaxed);
  }

  void set(int64_t value) noexcept {
    gauge.store(value, std::memory_order_relaxed);
  }

  int64_t read() const noexcept {
    return gauge.load(std::memory_order_relaxed);
  }
};

namespace HistogramDetail {
constexpr uint64_t doubleToBits(double value) noexcept {
  return std::bit_cast<uint64_t>(value);
}

constexpr double bitsToDouble(uint64_t bits) noexcept {
  return std::bit_cast<double>(bits);
}
} // namespace HistogramDetail

// Histograms maintain a count of data points accumulated since the last
// read() call, as well as extrema and sum of all data points. The state
// is reset when read.
//
// record() takes no lock: every field is updated with relaxed atomics, so
// request/step threads never serialise on this instrument. read() resets the
// window by exchanging each field back to its empty value, so it is not
// linearisable with respect to concurrent record() calls: a sample racing with
// read() may be dropped entirely, or split so that its value appears in one
// window and its count in the next. The guarantees are per-field atomicity,
// that no sample is counted twice, and that a quiescent instrument reports
// exactly what a lock-protected one would.
class Histogram {
public:
  struct Snapshot {
    uint64_t count = 0;
    double sum = 0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();
  };

  void record(double value) noexcept {
    count.fetch_add(1, std::memory_order_relaxed);
    addToSum(value);
    // Both CAS loops retry until they install a strictly smaller (larger)
    // value, so the final state is the minimum (maximum) of the values that
    // completed the loop. NaN compares false against everything, so it is
    // ignored here, matching the previous comparison-based implementation.
    uint64_t old = min.load(std::memory_order_relaxed);
    while (HistogramDetail::bitsToDouble(old) > value)
      if (min.compare_exchange_weak(old, HistogramDetail::doubleToBits(value),
                                    std::memory_order_relaxed))
        break;
    old = max.load(std::memory_order_relaxed);
    while (HistogramDetail::bitsToDouble(old) < value)
      if (max.compare_exchange_weak(old, HistogramDetail::doubleToBits(value),
                                    std::memory_order_relaxed))
        break;
  }

  Snapshot read() noexcept {
    Snapshot retVal;
    retVal.count = count.exchange(0, std::memory_order_relaxed);
    retVal.sum = HistogramDetail::bitsToDouble(
        sum.exchange(0, std::memory_order_relaxed));
    retVal.min = HistogramDetail::bitsToDouble(
        min.exchange(kMaxBits, std::memory_order_relaxed));
    retVal.max = HistogramDetail::bitsToDouble(
        max.exchange(kLowestBits, std::memory_order_relaxed));
    return retVal;
  }

private:
  // sum holds the bit pattern of the double accumulator; a floating-point
  // atomic has no fetch_add, so accumulate with a CAS loop.
  void addToSum(double value) noexcept {
    uint64_t old = sum.load(std::memory_order_relaxed);
    while (!sum.compare_exchange_weak(
        old,
        HistogramDetail::doubleToBits(HistogramDetail::bitsToDouble(old) +
                                      value),
        std::memory_order_relaxed))
      ;
  }

  static constexpr uint64_t kMaxBits =
      HistogramDetail::doubleToBits(std::numeric_limits<double>::max());
  static constexpr uint64_t kLowestBits =
      HistogramDetail::doubleToBits(std::numeric_limits<double>::lowest());

  std::atomic<uint64_t> count = 0;
  std::atomic<uint64_t> sum = 0;
  std::atomic<uint64_t> min = kMaxBits;
  std::atomic<uint64_t> max = kLowestBits;
};

// This struct is created by dumping the contents of the metric maps and is
// used by the Nanobind interface.
struct CollatedMetrics {
  struct CounterSample {
    llvm::StringRef name;
    uint64_t value;
  };
  struct GaugeSample {
    llvm::StringRef name;
    int64_t value;
  };
  struct HistogramSample {
    llvm::StringRef name;
    Histogram::Snapshot ss;
  };

  std::vector<CounterSample> counters;
  std::vector<GaugeSample> gauges;
  std::vector<HistogramSample> histograms;
};

// Registers metrics by name, returning the new metric to the caller.
// Responds to requests for new data snapshots by returning the
// contents of the maps.
class MetricsCollector {
  std::mutex mu;
  llvm::StringMap<Counter> counters;
  llvm::StringMap<Gauge> gauges;
  llvm::StringMap<Histogram> histograms;

public:
  CollatedMetrics collect();

  Counter &registerCounter(llvm::StringRef name);
  Gauge &registerGauge(llvm::StringRef name);
  Histogram &registerHistogram(llvm::StringRef name);
};

// The MetricsCollector is a singleton, retrieved through this helper.
MODULAR_VISIBILITY_EXPORT inline MetricsCollector &metrics() {
  // We need to be careful that the metrics collector isn't destroyed while
  // others hold references to its pointees. The OS will reclaim the memory.
  [[clang::no_destroy]] static MetricsCollector metrics;
  return metrics;
}

} // namespace M::Metrics

#endif // SUPPORT_METRICS_H
