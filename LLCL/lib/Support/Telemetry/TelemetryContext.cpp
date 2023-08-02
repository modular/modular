//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/Telemetry/Telemetry.h"

#include "LLCL/Support/Telemetry/Exporters/FileLogExporter.h"
#include "LLCL/Support/Telemetry/Exporters/FileMetricExporter.h"
#include "LLCL/Support/Telemetry/MetricReader.h"
#include "Support/Configuration.h"
#include "Support/Host.h"
#include <mutex>

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
#include "opentelemetry/sdk/resource/resource.h"
#endif // MODULAR_ENABLE_TELEMETRY

using namespace M;
using namespace LLCL;
using namespace Telemetry;
using namespace Exporter;

TelemetryContext::TelemetryContext(
    const llvm::StringMap<TelemetryContext::AttributeValue> &resources) {
#ifdef MODULAR_ENABLE_TELEMETRY
  using namespace opentelemetry::sdk::resource;

  // -------- Resources --------
  // Get the map of resources for the full host info.
  ResourceAttributes attrs;
  auto hostInfoOr = getHostMachineInfo();
  assert(!hostInfoOr.isError() && "could not get the host machine info");
  // Set the CPU and architecture.
  attrs.SetAttribute("cpu", hostInfoOr->cpuModelName);
  attrs.SetAttribute("arch", hostInfoOr->cpuArch);
  // Set the CPU features.
  std::vector<std::string_view> featuresView;
  for (auto &f : hostInfoOr->cpuFeatures)
    featuresView.emplace_back(f);
  attrs.SetAttribute("features", featuresView);
  // Set some of the other useful features, like number of cores and operating
  // system.
  attrs.SetAttribute("cores", hostInfoOr->numPhysicalCores);
  attrs.SetAttribute("operating system", hostInfoOr->osName);

  // Set the values of any resources we've been provided.
  for (auto &resource : resources) {
    std::visit([&](auto v) { attrs.SetAttribute(resource.first(), v); },
               resource.second);
  }

  // -------- Get config --------
  auto configOr = Config::open();
  if (configOr.isError())
    llvm::report_fatal_error(configOr.getError());
  Config cfg = std::move(*configOr);

  // Get the user ID out of the config.
  StringRef uuid = cfg.getValue("user.id");
  if (!uuid.empty())
    attrs.SetAttribute("userid", uuid);

  // Get the resource object we can give to OTel.
  auto otelResources = Resource::Create(attrs).Merge(Resource::GetDefault());

  // -------- Metrics --------
  // Initialize the MeterProvider.
  auto provider = std::make_unique<opentelemetry::sdk::metrics::MeterProvider>(
      std::make_unique<opentelemetry::sdk::metrics::ViewRegistry>(),
      otelResources);

  // Get metrics exporter config.
  StringRef httpEndpoint =
      cfg.getValue("telemetry.exporters.metrics.http_endpoint");
  std::filesystem::path filePath =
      cfg.getValue("telemetry.exporters.metrics.file_path").str();
  if (httpEndpoint.empty() && filePath.empty()) {
    // If no config provided, export to a default path.
    filePath = Config::getModularHomeDirPath() / "telemetry.log";
  }

  // Create metric readers, one for each exporter.

  if (!filePath.empty()) {
    // File exporter.
    auto exporter = std::make_unique<FileMetricExporter>(filePath);
    metricReaders.emplace_back(
        std::make_shared<ManualExportingMetricReader>(std::move(exporter)));
    provider->AddMetricReader(metricReaders.back());
  }

  if (!httpEndpoint.empty()) {
    // HTTP OTLP exporter.
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions otlpOptions;
    otlpOptions.url = (httpEndpoint + "/v1/metrics").str();
    auto exporter =
        opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(
            otlpOptions);
    metricReaders.emplace_back(
        std::make_shared<ManualExportingMetricReader>(std::move(exporter)));
    provider->AddMetricReader(metricReaders.back());
  }

  metricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      provider.release());
  meter = metricsProvider->GetMeter("modular");

  // -------- Logs --------
  // Get logs exporter config.
  httpEndpoint = cfg.getValue("telemetry.exporters.logs.http_endpoint");
  filePath = cfg.getValue("telemetry.exporters.logs.file_path").str();
  if (httpEndpoint.empty() && filePath.empty()) {
    // If no config provided, export to a default path.
    filePath = Config::getModularHomeDirPath() / "telemetry.log";
  }

  // Create log processors for each exporter.
  std::vector<std::unique_ptr<opentelemetry::sdk::logs::LogRecordProcessor>>
      processors;

  if (!filePath.empty()) {
    // File exporter.
    auto logExporter = std::make_unique<FileLogExporter>(filePath);
    processors.emplace_back(
        opentelemetry::sdk::logs::SimpleLogRecordProcessorFactory::Create(
            std::move(logExporter)));
  }

  if (!httpEndpoint.empty()) {
    // HTTP OTLP exporter.
    opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterOptions
        oltpLogOptions;
    oltpLogOptions.url = (httpEndpoint + "/v1/logs").str();
    auto logExporter =
        opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterFactory::Create(
            oltpLogOptions);
    processors.emplace_back(
        opentelemetry::sdk::logs::SimpleLogRecordProcessorFactory::Create(
            std::move(logExporter)));
  }

  loggerProvider = opentelemetry::sdk::logs::LoggerProviderFactory::Create(
      std::move(processors), otelResources);
  eventLoggerProvider =
      opentelemetry::sdk::logs::EventLoggerProviderFactory::Create();
#endif // MODULAR_ENABLE_TELEMETRY
}

TelemetryContext::~TelemetryContext() { flush(); }

void TelemetryContext::flush() {
#ifdef MODULAR_ENABLE_TELEMETRY
  // From OTel: Export must not be called concurrently for the same exporter
  // instance (collectAndExport calls Export).
  std::lock_guard<std::mutex> lock(exportLock);
  for (auto reader : metricReaders)
    reader->collectAndExport();
#endif // MODULAR_ENABLE_TELEMETRY
}
