//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/Telemetry.h"

#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/meter.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#endif // MODULAR_ENABLE_TELEMETRY

namespace M::Telemetry {

TelemetryContext::TelemetryContext() {
#ifdef MODULAR_ENABLE_TELEMETRY
  // Create OpenTelemetry OTLP HTTP exporter.
  opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions otlpOptions;
  otlpOptions.url = "http://localhost:4318/v1/metrics";
  auto exporter =
      opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(
          otlpOptions);

  // Initialize the MeterProvider.
  opentelemetry::sdk::metrics::PeriodicExportingMetricReaderOptions options;
  options.export_interval_millis = std::chrono::milliseconds(10000);
  options.export_timeout_millis = std::chrono::milliseconds(100);
  auto reader = std::make_shared<
      opentelemetry::sdk::metrics::PeriodicExportingMetricReader>(
      std::move(exporter), options);

  auto provider =
      std::make_unique<opentelemetry::sdk::metrics::MeterProvider>();
  provider->AddMetricReader(reader);
  metricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      provider.release());

  meter = metricsProvider->GetMeter("modular");

#endif // MODULAR_ENABLE_TELEMETRY
}

} // namespace M::Telemetry
