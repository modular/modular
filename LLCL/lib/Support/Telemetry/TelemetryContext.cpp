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
  }
  // Get the user ID out of the config.
  StringRef uuid = cfg.getValue("user.id");
  if (!uuid.empty())
    attrs.SetAttribute("userid", uuid);

  // Get the resource object we can give to OTel.
  auto otelResources = Resource::Create(attrs).Merge(Resource::GetDefault());

  // -------- Metrics --------
  // Create OpenTelemetry OTLP HTTP exporter.
  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter>
      metricExporter;
  if (httpEndpoint.empty()) {
    metricExporter = std::make_unique<FileMetricExporter>(filePath);
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

  auto provider = std::make_unique<opentelemetry::sdk::metrics::MeterProvider>(
      std::make_unique<opentelemetry::sdk::metrics::ViewRegistry>(),
      otelResources);
  provider->AddMetricReader(metricReader);
  metricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      provider.release());

  meter = metricsProvider->GetMeter("modular");

  // -------- Logs --------
  std::unique_ptr<opentelemetry::sdk::logs::LogRecordExporter> logExporter;
  if (httpEndpoint.empty()) {
    logExporter = std::make_unique<FileLogExporter>(filePath);
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
      std::move(processor), otelResources);
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
  metricReader->collectAndExport();
#endif // MODULAR_ENABLE_TELEMETRY
}
