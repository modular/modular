//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/Context.h"

M::TelemetryContext::TelemetryContext(bool enabled) {
  namespace MetricsSdk = opentelemetry::sdk::metrics;

  // TODO: currently only supporting `std::clog`, eventually this support
  // different types of exporters as provided by callers.
  if (enabled) {
    auto exporter = std::make_unique<
        opentelemetry::exporter::metrics::OStreamMetricExporter>(std::clog);

    MetricsSdk::PeriodicExportingMetricReaderOptions options;
    auto reader = std::make_unique<MetricsSdk::PeriodicExportingMetricReader>(
        std::move(exporter), options);
    metricsProvider =
        std::make_unique<opentelemetry::sdk::metrics::MeterProvider>();
    metricsProvider->AddMetricReader(std::move(reader));
  }
}

opentelemetry::sdk::metrics::MeterProvider &
M::TelemetryContext::getMetricsProvider() const {
  return *metricsProvider;
}
