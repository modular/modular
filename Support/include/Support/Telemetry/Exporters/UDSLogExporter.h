//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_EXPORTERS_UDSLOGEXPORTER_H
#define SUPPORT_TELEMETRY_EXPORTERS_UDSLOGEXPORTER_H

#include "Support/HTTP/HTTPClient.h"
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/exporters/otlp/otlp_http_client.h"
#include "opentelemetry/sdk/logs/exporter.h"
#include "opentelemetry/sdk/logs/recordable.h"

#endif // MODULAR_ENABLE_TELEMETRY

namespace M::Telemetry::Exporter {

#ifdef MODULAR_ENABLE_TELEMETRY

/// The FileLogExporter exports log data to a file, leveraging
/// OTel's OStreamLogRecordExporter.
class UDSLogExporter : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  explicit UDSLogExporter(const std::string &socketName,
                          const std::string &logsUrl = "http://v1/logs/")
      : socketName(socketName), logsUrl(logsUrl), client(HTTPContext::init()) {}

  /// Note that we need to hide the protobuf by patching otel with a function
  /// that returns a recordable instance
  std::unique_ptr<opentelemetry::sdk::logs::Recordable>
  MakeRecordable() noexcept override {
    // guard if telemetry enable?
    return opentelemetry::v1::exporter::otlp::MakeOtlpLogRecordable();
  }

  /// Exports a span of logs sent from the processor to a file.
  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::nostd::span<
         std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
             &records) noexcept override {

    auto protoBin =
        opentelemetry::v1::exporter::otlp::ConvertGenericMessageToProtoExported(
            records);
    HTTPRequest req{logsUrl};
    req.udsName = socketName;
    req.method = HTTPRequest::POST;
    req.bodyLen = protoBin.size();
    req.body = ContainerReadCallbackAdaptor(protoBin);
    auto response = client.executeRequest(req, llvm::outs());
    if (response.kind != HTTPResponse::Kind::Success) {
      llvm::errs() << "failed to export log via UDS"
                   << response.asError().getError() << "\n";
      return opentelemetry::sdk::common::ExportResult::kFailure;
    }

    return opentelemetry::sdk::common::ExportResult::kSuccess;
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
  std::string socketName;
  std::string logsUrl;
  HTTPClient client;
};

#endif // MODULAR_ENABLE_TELEMETRY

} // namespace M::Telemetry::Exporter

#endif // SUPPORT_TELEMETRY_EXPORTERS_UDSLOGEXPORTER_H
