//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/Telemetry/Telemetry.h"

#include "LLCL/Support/Telemetry/Exporters/FileLogExporter.h"
#include "LLCL/Support/Telemetry/Exporters/FileMetricExporter.h"
#include "Support/Configuration.h"
#include "Support/Host.h"
#include "llvm/Support/Threading.h"

#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_log_record_exporter_options.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/sdk/common/global_log_handler.h"
#include "opentelemetry/sdk/logs/event_logger_provider_factory.h"
#include "opentelemetry/sdk/logs/logger_provider.h"
#include "opentelemetry/sdk/logs/logger_provider_factory.h"
#include "opentelemetry/sdk/logs/processor.h"
#include "opentelemetry/sdk/logs/simple_log_record_processor_factory.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/meter.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/resource/resource.h"
#endif // MODULAR_ENABLE_TELEMETRY

#define DEBUG_TYPE "telemetry-context"

using namespace M;
using namespace LLCL;
using namespace Telemetry;
using namespace Exporter;

#ifdef MODULAR_ENABLE_TELEMETRY
static void configureInternalLogging(Config &cfg) {
  // OTel internal logging (e.g. warnings and errors related to OTel's
  // operation) is off by default and controlled with `telemetry.internal_log`
  // config key (or equivalently with `TELEMETRY_INTERNAL_LOG` env var).
  bool internalLogsOff = false;
  opentelemetry::sdk::common::internal_log::LogLevel logLevel;
  StringRef internalLogConfig = cfg.getValue("telemetry.internal_log");
  if (internalLogConfig.empty() || internalLogConfig == "off") {
    internalLogsOff = true;
  } else {
    if (internalLogConfig == "error") {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Error;
    } else if (internalLogConfig.startswith("warn")) {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Warning;
    } else if (internalLogConfig == "info") {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Info;
    } else if (internalLogConfig == "debug") {
      logLevel = opentelemetry::sdk::common::internal_log::LogLevel::Debug;
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Unrecognized log level for telemetry.internal_log: "
                 << internalLogConfig);
      internalLogsOff = true;
    }
  }
  if (internalLogsOff) {
    // Use NOOP log handler to disable all OTel internal logs.
    auto noopHandler = std::make_shared<
        opentelemetry::sdk::common::internal_log::NoopLogHandler>();
    opentelemetry::sdk::common::internal_log::GlobalLogHandler::SetLogHandler(
        noopHandler);
  } else {
    opentelemetry::sdk::common::internal_log::GlobalLogHandler::SetLogLevel(
        logLevel);
  }
}
#endif // MODULAR_ENABLE_TELEMETRY

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

  // Check if telemetry is enabled. Note that currently users have to opt out of
  // telemetry, so it is enabled unless the user explicitly disables.
  bool enabled = true;
  if (cfg.getValue("telemetry.enabled").equals_insensitive("false"))
    enabled = false;

  // Configure OTel internal logging.
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() { configureInternalLogging(cfg); });

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

  opentelemetry::sdk::metrics::PeriodicExportingMetricReaderOptions options;
  options.export_interval_millis = kExportInterval;
  options.export_timeout_millis = kExportTimeout;

  // Get metrics exporter config.
  StringRef httpEndpoint =
      cfg.getValue("telemetry.exporters.metrics.http_endpoint");
  std::filesystem::path filePath =
      cfg.getValue("telemetry.exporters.metrics.file_path").str();
  if (enabled && httpEndpoint.empty() && filePath.empty()) {
    // If telemetry is enabled and no config is provided, export to our default
    // URL.
    httpEndpoint = kTelemetryUrl;
  }

  // Create metric readers, one for each exporter.

  if (enabled && !filePath.empty()) {
    // File exporter.
    auto exporter = std::make_unique<FileMetricExporter>(filePath);
    auto reader = std::make_shared<
        opentelemetry::sdk::metrics::PeriodicExportingMetricReader>(
        std::move(exporter), options);
    provider->AddMetricReader(reader);
  }

  if (enabled && !httpEndpoint.empty()) {
    // HTTP OTLP exporter.
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions otlpOptions;
    otlpOptions.url = (httpEndpoint + "/v1/metrics").str();
    auto exporter =
        opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(
            otlpOptions);
    auto reader = std::make_shared<
        opentelemetry::sdk::metrics::PeriodicExportingMetricReader>(
        std::move(exporter), options);
    provider->AddMetricReader(reader);
  }

  metricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      provider.release());
  meter = metricsProvider->GetMeter("modular");

  // -------- Logs --------
  // Get logs exporter config.
  httpEndpoint = cfg.getValue("telemetry.exporters.logs.http_endpoint");
  filePath = cfg.getValue("telemetry.exporters.logs.file_path").str();
  if (enabled && httpEndpoint.empty() && filePath.empty()) {
    // If telemetry is enabled and no config is provided, export to our default
    // URL.
    httpEndpoint = kTelemetryUrl;
  }

  // Create log processors for each exporter.
  std::vector<std::unique_ptr<opentelemetry::sdk::logs::LogRecordProcessor>>
      processors;

  if (enabled && !filePath.empty()) {
    // File exporter.
    auto logExporter = std::make_unique<FileLogExporter>(filePath);
    processors.emplace_back(
        opentelemetry::sdk::logs::SimpleLogRecordProcessorFactory::Create(
            std::move(logExporter)));
  }

  if (enabled && !httpEndpoint.empty()) {
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

TelemetryContext::~TelemetryContext() { flush(kShutdownFlushTimeout); }

void TelemetryContext::flush(std::chrono::microseconds timeout) {
#ifdef MODULAR_ENABLE_TELEMETRY
  // Flush metrics.
  auto metricsProviderImpl =
      static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
          metricsProvider.get());
  metricsProviderImpl->ForceFlush(timeout);

  // Flush logs.
  auto loggerProviderImpl =
      std::static_pointer_cast<opentelemetry::sdk::logs::LoggerProvider>(
          loggerProvider);
  loggerProviderImpl->ForceFlush(timeout);
#endif // MODULAR_ENABLE_TELEMETRY
}
