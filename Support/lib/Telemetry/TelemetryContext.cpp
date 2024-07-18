//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/Exporters/UDSLogExporter.h"
#include "Support/Telemetry/Exporters/UDSMetricExporter.h"
#include "Support/Telemetry/Telemetry.h"

#include "Config/Version.h"
#include "Support/Base64.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/FileSystemExtras.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Random.h"
#include "Support/Settings/Settings.h"
#include "Support/Telemetry/Exporters/FileLogExporter.h"
#include "Support/Telemetry/Exporters/FileMetricExporter.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/Support/BLAKE3.h"
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
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/metrics/meter_provider_factory.h"

#include "opentelemetry/sdk/logs/processor.h"
#include "opentelemetry/sdk/logs/simple_log_record_processor_factory.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/meter.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/resource/resource.h"
#endif // MODULAR_ENABLE_TELEMETRY

// enable TEST_UDS to use unix domain sockets for log/metrics. Do set the config
// value for `telemetry.exporters.metrics.uds_name` to the right socket from the
// server
#define TEST_UDS 0

#include <algorithm> // For std::sort.

#define DEBUG_TYPE "telemetry-context"

using namespace M;
using namespace Telemetry;
using namespace Exporter;

#ifdef MODULAR_ENABLE_TELEMETRY
static Level levelFromString(StringRef levelStr) {
  if (levelStr.empty())
    return Level::L1;

  int level = 0;
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

static void configureInternalLogging(StringRef internalLogConfig) {
  // OTel internal logging (e.g. warnings and errors related to OTel's
  // operation) is off by default and controlled with `telemetry.internal_log`
  // config key (or equivalently with `TELEMETRY_INTERNAL_LOG` env var).
  bool internalLogsOff = false;
  opentelemetry::sdk::common::internal_log::LogLevel logLevel;
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

#ifdef AF_PACKET
#define __AF_TYPE AF_PACKET
#else
#define __AF_TYPE AF_LINK
#endif

/// creates local identifiers; see Telemetry.h.
std::pair<std::string, std::string> M::Telemetry::createLocalIDs() {
  // Collect all interfaces.
  std::vector<std::string> macs = localMACs();
  std::sort(std::begin(macs), std::end(macs));

  // Hash all addresses together.
  llvm::BLAKE3 hashState{};
  for (const auto &mac : macs)
    hashState.update(StringRef(mac));

  // Construct a machine ID.
  auto hash = hashState.result();
  std::string machine_id =
      encodeURLSafeBase64(std::string({hash.begin(), hash.end()}));

  // Mix in some random bytes in order to construct a local session identifier.
  // This may suffer from a cardinality explosion (and we may choose to rely on
  // the machineid in the future instead), but we can make that decision in the
  // backend separately.
  SecureRandomBytesGenerator rng;
  std::array<uint8_t, 32> scratchBuf = {};
  auto err = rng.getRandomBytes(scratchBuf);
  assert(!err.isError());
  hashState.update(scratchBuf);
  hash = hashState.result();
  std::string session_id =
      encodeURLSafeBase64(std::string({hash.begin(), hash.end()}));

  // Return the pair.
  return std::pair<std::string, std::string>(machine_id, session_id);
}

static size_t getMaxProcessors(const HostMachineInfo &hostInfo) {
  auto limitsOr = CPULimits::get();
  if (!limitsOr.isError()) {
    auto millicores = limitsOr->millicores;
    if (millicores.has_value())
      return *millicores / 1000;
  }
  return hostInfo.numPhysicalCores;
}
#endif // MODULAR_ENABLE_TELEMETRY

TelemetryContext::~TelemetryContext() {
  flush(kShutdownFlushTimeout);
#ifdef MODULAR_ENABLE_TELEMETRY
  // Flush metrics.
  auto metricsProviderImpl =
      static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
          metricsProvider.get());
  metricsProviderImpl->Shutdown();
  if (userMetricsProvider) {
    auto userMetricsProviderImpl =
        static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
            userMetricsProvider.get());
    userMetricsProviderImpl->Shutdown();
  }

  // Flush logs.
  auto loggerProviderImpl =
      std::static_pointer_cast<opentelemetry::sdk::logs::LoggerProvider>(
          loggerProvider);
  loggerProviderImpl->Shutdown();
#endif // MODULAR_ENABLE_TELEMETRY
}

void TelemetryContext::flush(std::chrono::microseconds timeout) {
#ifdef MODULAR_ENABLE_TELEMETRY
  // Flush metrics.
  auto metricsProviderImpl =
      static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
          metricsProvider.get());
  metricsProviderImpl->ForceFlush(timeout);
  if (userMetricsProvider) {
    auto userMetricsProviderImpl =
        static_cast<opentelemetry::sdk::metrics::MeterProvider *>(
            userMetricsProvider.get());
    userMetricsProviderImpl->ForceFlush(timeout);
  }

  // Flush logs.
  auto loggerProviderImpl =
      std::static_pointer_cast<opentelemetry::sdk::logs::LoggerProvider>(
          loggerProvider);
  loggerProviderImpl->ForceFlush(timeout);
#endif // MODULAR_ENABLE_TELEMETRY
}

TelemetryContext::TelemetryContext(
    Settings &settings, const llvm::StringMap<AttributeValue> &resources) {
  [[maybe_unused]] bool isProdBuild = false;
#ifdef MODULAR_PRODUCTION
  isProdBuild = true;
#endif
#ifdef MODULAR_ENABLE_TELEMETRY
  using namespace opentelemetry::sdk::resource;
  // -------- Resources --------
  // Get the map of resources for the full host info.
  ResourceAttributes attrs;
  auto hostInfoOr = getHostMachineInfo();
  assert(!hostInfoOr.isError() && "could not get the host machine info");
  // Set the CPU and architecture.
  attrs.SetAttribute("cpu.description", hostInfoOr->cpuModelName);
  // WARNING: Metering & billing depends on cpu.arch. Do not remove!
  attrs.SetAttribute("cpu.arch", hostInfoOr->cpuArch);
  // Set the CPU features.
  std::vector<std::string_view> featuresView;
  for (auto &f : hostInfoOr->cpuFeatures)
    featuresView.emplace_back(f);
  attrs.SetAttribute("cpu.features", featuresView);
  // Set some of the other useful features, like number of cores and operating
  // system.
  attrs.SetAttribute("cpu.cores", hostInfoOr->numPhysicalCores);
  attrs.SetAttribute("cpu.max_cores", getMaxProcessors(*hostInfoOr));
  attrs.SetAttribute("cpu.model_name", hostInfoOr->cpuModelName);
  attrs.SetAttribute("os.type", hostInfoOr->osName);
  attrs.SetAttribute("os.version", hostInfoOr->osVersion);

  // Get total memory.
  auto memoryOr = getHostTotalMemoryKB();
  if (!memoryOr.isError()) {
    attrs.SetAttribute("memory", memoryOr.takeValue());
  }

  // Check if we are running in a container
  auto isInContainer = getHostIsInContainer();
  if (!isInContainer.isError()) {
    attrs.SetAttribute("in.container", isInContainer.takeValue());
  }

  // Set the underlying Modular version.
  auto version = getModularVersion();
  attrs.SetAttribute("modular.version.major", version.major);
  attrs.SetAttribute("modular.version.minor", version.minor);
  attrs.SetAttribute("modular.version.patch", version.patch);
  attrs.SetAttribute("modular.version.label", version.label);
  attrs.SetAttribute("modular.version.revision", version.revision);
  attrs.SetAttribute("modular.version.buildtype", version.buildType);

  // Set the local machineid.
  static std::pair<std::string, std::string> local_ids = createLocalIDs();
  // WARNING: Metering & billing depends on machineid. Do not remove!
  attrs.SetAttribute("machineid", local_ids.first);
  attrs.SetAttribute("sessionid", local_ids.second);
  machineId = local_ids.first;

  auto webId = dyn_cast_if_present<StringRef>(settings.get("web.id"));
  if (!webId.empty()) {
    attrs.SetAttribute("web.user.id", webId);
  }

  // Set the values of any resources we've been provided.
  for (auto &resource : resources) {
    std::visit([&](auto v) { attrs.SetAttribute(resource.first(), v); },
               resource.second);
  }

  // Check if telemetry is enabled. Note that currently users have to opt out
  // of telemetry, so it is enabled unless the user explicitly disables.
  bool enabled = settings.getBool("telemetry.enabled", true);

  // Get telemetry level.
  auto *cfgLevel = settings.get("telemetry.level");
  auto level = llvm::dyn_cast_if_present<StringRef>(cfgLevel);
  telemetryLevel = levelFromString(level);

  // Fix the minimum telemetry level for a non-modular developer to 0. This can
  // be changed to use a different entitlement in the future.
  if (!enabled && !settings.getBool<ModularDeveloperEntitlement>()) {
    enabled = true;
    telemetryLevel = Level::L0;
  }
  // Configure OTel internal logging.
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() {
    configureInternalLogging(
        dyn_cast_if_present<StringRef>(settings.get("telemetry.internal_log")));
  });

  // Get the user ID if we have one.
  attrs.SetAttribute("enduser.id", settings.get<StringRef>("user.id"));

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
  auto httpEndpoint =
      settings.get<StringRef>("telemetry.exporters.metrics.http_endpoint");
  std::filesystem::path filePath =
      settings.get<StringRef>("telemetry.exporters.metrics.file_path").str();
  std::filesystem::path udsName =
      settings.get<StringRef>("telemetry.exporters.metrics.uds_name").str();

  // Only allow modular developers to overwrite this endpoint.
  if (isProdBuild && !settings.getBool<ModularDeveloperEntitlement>())
    httpEndpoint = kTelemetryUrl;

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
#if TEST_UDS
    auto exporter = std::make_unique<UDSMetricExporter>(udsName);
#else

    // HTTP OTLP exporter.
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions otlpOptions;
    otlpOptions.ssl_client_key_path = settings.clientKeyPriv();
    otlpOptions.ssl_client_cert_path = settings.clientCert();

    otlpOptions.url = (httpEndpoint + "/v1/metrics").str();
    auto exporter =
        opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(
            otlpOptions);
#endif
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
  httpEndpoint =
      settings.get<StringRef>("telemetry.exporters.logs.http_endpoint");
  filePath =
      settings.get<StringRef>("telemetry.exporters.logs.file_path").str();

  // See above; only developers can overwrite the httpEndpoint.
  if (isProdBuild && !settings.getBool<ModularDeveloperEntitlement>())
    httpEndpoint = kTelemetryUrl;

  // Create log processors for each exporter.
  std::vector<std::unique_ptr<opentelemetry::sdk::logs::LogRecordProcessor>>
      processors;

  if (enabled && !filePath.empty()) {

    auto logExporter = std::make_unique<FileLogExporter>(filePath);
    processors.emplace_back(
        opentelemetry::sdk::logs::SimpleLogRecordProcessorFactory::Create(
            std::move(logExporter)));
  }

  if (enabled && !httpEndpoint.empty()) {
#if TEST_UDS
    auto logExporter = std::make_unique<UDSLogExporter>(udsName, "/v1/logs");
#else

    // HTTP OTLP exporter.
    opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterOptions
        otlpLogOptions;
    otlpLogOptions.ssl_client_key_string = settings.clientKeyPriv();
    otlpLogOptions.ssl_client_cert_string = settings.clientCert();

    otlpLogOptions.url = (httpEndpoint + "/v1/logs").str();
    auto logExporter =
        opentelemetry::exporter::otlp::OtlpHttpLogRecordExporterFactory::Create(
            otlpLogOptions);
#endif
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

#ifdef MODULAR_ENABLE_TELEMETRY
bool TelemetryContext::initUserMetricsReader(
    std::unique_ptr<opentelemetry::sdk::metrics::MetricReader> reader) {
  if (userMetricsProvider) {
    llvm::dbgs() << "Custom metric provider already set\n";
    return false;
  }
  auto userReader = std::shared_ptr<opentelemetry::sdk::metrics::MetricReader>(
      reader.release());
  assert(userReader.get() && "user metrics reader is null");
  auto userProvider =
      std::make_unique<opentelemetry::sdk::metrics::MeterProvider>(
          // TODO - SERV-103 - what attributes to put here?
          //  std::make_unique<opentelemetry::sdk::metrics::ViewRegistry>(),
          //  otelResources
      );

  userProvider->AddMetricReader(userReader);
  userMetricsProvider = std::unique_ptr<opentelemetry::metrics::MeterProvider>(
      userProvider.release());
  userMeter = userMetricsProvider->GetMeter("max_serve");
  return true;
}
#endif

bool TelemetryContext::clearUserMetricsReader() {
#ifdef MODULAR_ENABLE_TELEMETRY
  if (!userMetricsProvider) {
    return false;
  }

  userMetricsProvider.reset();
  userMeter.reset();
  return true;
#else
  return false;
#endif
}
