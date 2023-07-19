//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/MetricReader.h"
#include "Support/Telemetry/Telemetry.h"

#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_options.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/logs/event_logger_provider_factory.h"
#include "opentelemetry/sdk/logs/logger_provider_factory.h"
#include "opentelemetry/sdk/logs/processor.h"
#include "opentelemetry/sdk/logs/simple_log_record_processor_factory.h"
#include "opentelemetry/sdk/metrics/meter.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#endif // MODULAR_ENABLE_TELEMETRY

namespace M::Telemetry {

TelemetryContext::TelemetryContext() {
#ifdef MODULAR_ENABLE_TELEMETRY

  // -------- Metrics --------
  // Create OpenTelemetry OTLP HTTP exporter.
  opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions otlpOptions;
  otlpOptions.url = "http://localhost:4318/v1/metrics";
  auto exporter =
      opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(
          otlpOptions);

  // Initialize the MeterProvider.
  metricReader =
      std::make_shared<ManualExportingMetricReader>(std::move(exporter));

  auto provider =
      std::make_unique<opentelemetry::sdk::metrics::MeterProvider>();
  provider->AddMetricReader(metricReader);
  metricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      provider.release());

  meter = metricsProvider->GetMeter("modular");

  // -------- Logs --------
  opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterOptions
      oltpLogOptions;
  oltpLogOptions.url = "http://localhost:4318/v1/logs";
  // Create OTLP exporter instance
  auto logExporter =
      opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterFactory::Create(
          oltpLogOptions);
  auto processor =
      opentelemetry::sdk::logs::SimpleLogRecordProcessorFactory::Create(
          std::move(logExporter));
  loggerProvider = opentelemetry::sdk::logs::LoggerProviderFactory::Create(
      std::move(processor));
  eventLoggerProvider =
      opentelemetry::sdk::logs::EventLoggerProviderFactory::Create();
#endif // MODULAR_ENABLE_TELEMETRY
}

TelemetryContext::~TelemetryContext() {
#ifdef MODULAR_ENABLE_TELEMETRY
  metricReader->collectAndExport();
#endif
}

} // namespace M::Telemetry
