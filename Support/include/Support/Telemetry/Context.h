//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef SUPPORT_TELEMETRY_CONTEXT_H
#define SUPPORT_TELEMETRY_CONTEXT_H

#include "opentelemetry/exporters/ostream/metric_exporter.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/meter.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/metrics/push_metric_exporter.h"

#include <memory>

namespace M {

class TelemetryContext {
public:
  /// Initializes metrics
  TelemetryContext();

  opentelemetry::sdk::metrics::MeterProvider &getMetricsProvider() const;

private:
  std::unique_ptr<opentelemetry::sdk::metrics::MeterProvider> metricsProvider;
};

} // namespace M

#endif // SUPPORT_TELEMETRY_CONTEXT_H
