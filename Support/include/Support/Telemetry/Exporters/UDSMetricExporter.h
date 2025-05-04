//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_EXPORTERS_UDSMetricExporter_H
#define SUPPORT_TELEMETRY_EXPORTERS_UDSMetricExporter_H

#include "Support/HTTP/HTTPClient.h"
#include "llvm/Support/Debug.h"
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter.h"
#include "opentelemetry/sdk/metrics/push_metric_exporter.h"

#endif // MODULAR_ENABLE_TELEMETRY

namespace M::Telemetry::Exporter {

#ifdef MODULAR_ENABLE_TELEMETRY

#define USE_NATIVE_SOCKET 0

class UDSMetricExporter final
    : public opentelemetry::sdk::metrics::PushMetricExporter {
public:
  explicit UDSMetricExporter(const std::string &socketName,
                             const std::string &metricsUrl = "http://metrics/")
      : socketName(socketName), metricsUrl(metricsUrl),
        client(HTTPContext::init()) {}
  UDSMetricExporter(UDSMetricExporter &) = delete;

  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::sdk::metrics::ResourceMetrics &data) noexcept
      override {

    auto protoBin =
        opentelemetry::v1::exporter::otlp::ConvertGenericMessageToProtoExported(
            data);
    HTTPRequest req{metricsUrl};
    req.udsName = socketName;
    req.method = HTTPRequest::POST;
    req.bodyLen = protoBin.size();

    req.body = ContainerReadCallbackAdaptor(protoBin);
    auto response = client.executeRequest(req, llvm::outs());
    if (response.kind != HTTPResponse::Kind::Success) {
      llvm::errs() << "failed to export metric via UDS"
                   << response.asError().getError() << "\n";
      return opentelemetry::sdk::common::ExportResult::kFailure;
    }

    return opentelemetry::sdk::common::ExportResult::kSuccess;
  }

  /// Get the AggregationTemporality for the exporter.
  opentelemetry::sdk::metrics::AggregationTemporality GetAggregationTemporality(
      opentelemetry::sdk::metrics::InstrumentType instrument_type)
      const noexcept override {
    return opentelemetry::v1::sdk::metrics::AggregationTemporality::
        kUnspecified;
  }

  /// Force flush the exporter.
  bool ForceFlush(std::chrono::microseconds timeout =
                      std::chrono::microseconds::max()) noexcept override {

    return true;
  }

  /// Shut down the exporter, with optional timeout.
  bool Shutdown(std::chrono::microseconds timeout =
                    std::chrono::microseconds::max()) noexcept override {

    return true;
  }

private:
  /// Logs are exported to this file.
  std::string socketName;
  std::string metricsUrl;
  HTTPClient client;
};

#endif // MODULAR_ENABLE_TELEMETRY

} // namespace M::Telemetry::Exporter

#endif // SUPPORT_TELEMETRY_EXPORTERS_UDSMetricExporter_H
