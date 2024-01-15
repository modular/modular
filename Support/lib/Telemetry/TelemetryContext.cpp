//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/Telemetry.h"

#include "Support/MArchTarget/Host.h"
#include "Support/Telemetry/Exporters/FileLogExporter.h"
#include "Support/Telemetry/Exporters/FileMetricExporter.h"
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
using namespace Telemetry;
using namespace Exporter;

#ifdef MODULAR_ENABLE_TELEMETRY
static Level levelFromString(StringRef levelStr) {
  int level;
  if (levelStr.getAsInteger(10, level))
    assert(false && "Non-integer telemetry level specified");
  assert((level >= 0 && level < 3) && "Telemetry level outside [0,2] range");
  if (level == 0)
    return Level::L0;
  if (level == 1)
    return Level::L1;
  if (level == 2)
    return Level::L2;
  llvm_unreachable("unknown telemetry level");
}

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
    } else if (internalLogConfig.starts_with("warn")) {
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
    const EntitlementStore &entitlementStore,
    const llvm::StringMap<TelemetryContext::AttributeValue> &resources,
    std::optional<Config> config) {
#ifdef MODULAR_ENABLE_TELEMETRY
  using namespace opentelemetry::sdk::resource;

  // -------- Resources --------
  // Get the map of resources for the full host info.
  ResourceAttributes attrs;
  auto hostInfoOr = getHostMachineInfo();
  assert(!hostInfoOr.isError() && "could not get the host machine info");
  // Set the CPU and architecture.
  attrs.SetAttribute("cpu.description", hostInfoOr->cpuModelName);
  attrs.SetAttribute("cpu.arch", hostInfoOr->cpuArch);
  // Set the CPU features.
  std::vector<std::string_view> featuresView;
  for (auto &f : hostInfoOr->cpuFeatures)
    featuresView.emplace_back(f);
  attrs.SetAttribute("cpu.features", featuresView);
  // Set some of the other useful features, like number of cores and operating
  // system.
  attrs.SetAttribute("cpu.cores", hostInfoOr->numPhysicalCores);
  attrs.SetAttribute("os.type", hostInfoOr->osName);

  // Set the values of any resources we've been provided.
  for (auto &resource : resources) {
    std::visit([&](auto v) { attrs.SetAttribute(resource.first(), v); },
               resource.second);
  }

  // -------- Get config --------
  Config cfg;
  if (config) {
    cfg = std::move(*config);
  } else {
    auto configOr = Config::open();
    if (configOr.isError())
      llvm::report_fatal_error(configOr.getError());
    cfg = std::move(*configOr);
  }

  // Check if telemetry is enabled. Note that currently users have to opt out of
  // telemetry, so it is enabled unless the user explicitly disables.
  bool enabled = true;
  if (cfg.getValue("telemetry.enabled").equals_insensitive("false"))
    enabled = false;

  // Get telemetry level.
  StringRef cfgLevel = cfg.getValue("telemetry.level");
  if (cfgLevel == "")
    telemetryLevel = Level::L0;
  else
    telemetryLevel = levelFromString(cfgLevel);

  // Configure OTel internal logging.
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() { configureInternalLogging(cfg); });

  // Get the user ID out of the EntitlementStore.
  auto store = EntitlementStore::alwaysOpen(nullptr, llvm::errs());
  auto userIDOr = store.getUserID(std::move(config));
  if (!userIDOr.isError())
    attrs.SetAttribute("enduser.id", *userIDOr);

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

  // Extend the histogram buckets for our timers. The default's max bucket is
  // 10000 ms.
  auto instrument_selector =
      std::make_unique<opentelemetry::sdk::metrics::InstrumentSelector>(
          opentelemetry::sdk::metrics::InstrumentType::kHistogram, ".*\\.time$",
          "ms");
  auto meter_selector =
      std::make_unique<opentelemetry::sdk::metrics::MeterSelector>("", "", "");
  auto histConfig = std::make_shared<
      opentelemetry::sdk::metrics::HistogramAggregationConfig>();
  histConfig->boundaries_ = {0,     50,    100,   250,   500,   750,   1000,
                             2500,  5000,  7500,  10000, 12500, 15000, 17500,
                             20000, 25000, 30000, 40000, 50000};
  auto view = std::make_unique<opentelemetry::sdk::metrics::View>(
      "", "", "", opentelemetry::sdk::metrics::AggregationType::kHistogram,
      histConfig);

  provider->AddView(std::move(instrument_selector), std::move(meter_selector),
                    std::move(view));

  // Get metrics exporter config.
  StringRef httpEndpoint =
      cfg.getValue("telemetry.exporters.metrics.http_endpoint");
  std::filesystem::path filePath =
      cfg.getValue("telemetry.exporters.metrics.file_path").str();
  if (httpEndpoint.empty() && filePath.empty()) {
    // If no config is provided, export to our default URL.
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

  noopMetricsProvider =
      std::make_unique<opentelemetry::metrics::NoopMeterProvider>();
  noopMeter = noopMetricsProvider->GetMeter("modular");

  // -------- Logs --------
  // Get logs exporter config.
  httpEndpoint = cfg.getValue("telemetry.exporters.logs.http_endpoint");
  filePath = cfg.getValue("telemetry.exporters.logs.file_path").str();
  if (httpEndpoint.empty() && filePath.empty()) {
    // If no config is provided, export to our default URL.
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
