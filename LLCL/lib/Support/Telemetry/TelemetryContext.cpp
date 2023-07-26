//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/Telemetry/Telemetry.h"

#include "LLCL/Support/Telemetry/MetricReader.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"
#include <filesystem>
#include <fstream>
#include <mutex>

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

#define DEBUG_TYPE "telemetry-context"

using namespace M;
using namespace LLCL;
using namespace Telemetry;

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
  filePath = cfg.getValue("telemetry.exporters.metrics.file_path").str();
  if (httpEndpoint.empty()) {
    // Default to MODULAR_HOME/telemetry.log
    if (filePath.empty())
      filePath = Config::getModularHomeDirPath() / "telemetry.log";
  }

  // -------- Metrics --------
  // Create OpenTelemetry OTLP HTTP exporter.
  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter>
      metricExporter;
  if (httpEndpoint.empty()) {
    metricExporter = std::make_unique<
        opentelemetry::exporter::metrics::OStreamMetricExporter>(outputStream);
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
        opentelemetry::exporter::logs::OStreamLogRecordExporter>(outputStream);
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

TelemetryContext::~TelemetryContext() { flush(); }

void TelemetryContext::flush() {
#ifdef MODULAR_ENABLE_TELEMETRY
  // From OTel: Export must not be called concurrently for the same exporter
  // instance (collectAndExport calls Export). This also conveniently holds the
  // export lock so we don't have multiple flushes attempting to mutate the put
  // pointer on the stream.
  std::lock_guard<std::mutex> lock(exportLock);
  metricReader->collectAndExport();

  if (filePath.empty())
    return;

  // Flush the stream to a file, if it exists.
  auto err = appendFileUnderLock(filePath, [&](llvm::raw_ostream &os) {
    // Do the stream manipulation inside the atomic region - other things may
    // try to write during this, and we need to hold the lock.
    os << outputStream.str();
    // Seek back to the beginning.
    outputStream.seekp(0, std::ios_base::beg);
  });
  if (err.isError())
    LLVM_DEBUG(llvm::dbgs() << err.getError());
#endif // MODULAR_ENABLE_TELEMETRY
}
