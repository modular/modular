//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Metrics.h"

namespace M::Metrics {

CollatedMetrics MetricsCollector::collect() {
  CollatedMetrics result;
  std::lock_guard<std::mutex> guard(mu);
  for (auto &e : counters)
    result.counters.push_back({e.getKey(), e.second.read()});
  for (auto &e : gauges)
    result.gauges.push_back({e.getKey(), e.second.read()});
  for (auto &e : histograms)
    result.histograms.push_back({e.getKey(), e.second.read()});
  return result;
}

Counter &MetricsCollector::registerCounter(llvm::StringRef name) {
  std::lock_guard<std::mutex> guard(mu);
  return counters.try_emplace(name).first->second;
}

Gauge &MetricsCollector::registerGauge(llvm::StringRef name) {
  std::lock_guard<std::mutex> guard(mu);
  return gauges.try_emplace(name).first->second;
}

Histogram &MetricsCollector::registerHistogram(llvm::StringRef name) {
  std::lock_guard<std::mutex> guard(mu);
  return histograms.try_emplace(name).first->second;
}

} // namespace M::Metrics
