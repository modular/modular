//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/Telemetry.h"

#include "Support/Configuration.h"
#include "Support/Telemetry/MetricReader.h"
#include "llvm/ADT/Twine.h"
#include <filesystem>
#include <fstream>

#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/exporters/ostream/log_record_exporter.h"
#include "opentelemetry/exporters/ostream/metric_exporter.h"
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

  // -------- Exporter options (export to file or to OTLP receiver) --------
  auto configOr = Config::open();
  if (configOr.isError())
    llvm::report_fatal_error(configOr.getError());
  Config cfg = std::move(*configOr);
  // TODO: Allow separate configuration for metrics and logs, and multiple
  //       exporters for each (if OTel allows it).
  StringRef httpEndpoint =
      cfg.getValue("telemetry.exporters.metrics.http_endpoint");
  std::filesystem::path filePath =
      cfg.getValue("telemetry.exporters.metrics.file_path").str();
  if (httpEndpoint.empty()) {
    // Default to MODULAR_HOME/telemetry.log
    if (filePath.empty())
      filePath = Config::getModularHomeDirPath() / "telemetry.log";

    outputFile = std::make_unique<std::ofstream>();
    outputFile->open(filePath, std::ios_base::app);
    assert(outputFile->is_open() && "could not open file for telemetry output");
  }

  // -------- Metrics --------
  // Create OpenTelemetry OTLP HTTP exporter.
  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter>
      metricExporter;
  if (httpEndpoint.empty()) {
    metricExporter = std::make_unique<
        opentelemetry::exporter::metrics::OStreamMetricExporter>(*outputFile);
  } else {
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions otlpOptions;
    otlpOptions.url = (httpEndpoint + "/v1/metrics").str();
    metricExporter =
        opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(
            otlpOptions);
  }
  // Initialize the MeterProvider.
  metricReader =
      std::make_shared<ManualExportingMetricReader>(std::move(metricExporter));

  auto provider =
      std::make_unique<opentelemetry::sdk::metrics::MeterProvider>();
  provider->AddMetricReader(metricReader);
  metricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      provider.release());

  meter = metricsProvider->GetMeter("modular");

  // -------- Logs --------
  std::unique_ptr<opentelemetry::sdk::logs::LogRecordExporter> logExporter;
  if (httpEndpoint.empty()) {
    logExporter = std::make_unique<
        opentelemetry::exporter::logs::OStreamLogRecordExporter>(*outputFile);
  } else {
    opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterOptions
        oltpLogOptions;
    oltpLogOptions.url = (httpEndpoint + "/v1/logs").str();
    // Create OTLP exporter instance
    logExporter =
        opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterFactory::Create(
            oltpLogOptions);
  }
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
