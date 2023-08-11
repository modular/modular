##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

if(MODULAR_ENABLE_TELEMETRY)
  if(${DISABLE_ALL_EXTERNAL})
    message(FATAL_ERROR "Cannot enable Telemetry since OpenTelemetry is not available.")
    return()
  endif()

  include(AddOpenTelemetry)

  add_modular_library(LLCLTelemetry
    TelemetryContext.cpp
    FileMetricExporter.cpp
    FileLogExporter.cpp

    LINK_LIBS PUBLIC
    LLCLSupport
    MSupport
    opentelemetry_metrics
    opentelemetry_logs

    # It's important to keep OTLP PRIVATE. If not, it will leak our protobuf headers
    # across the codebase given that LLCL links with MTelemetry, and stock TF headers
    # will throw a build-time error because they were generated with an older protobuf.
    LINK_LIBS PRIVATE
    opentelemetry_exporter_otlp_http
    opentelemetry_exporter_otlp_http_metric
    opentelemetry_exporter_otlp_http_log
    opentelemetry_resources
    opentelemetry_exporter_ostream_metrics
    opentelemetry_exporter_ostream_logs

    DEPENDS protobuf::protoc
    )

else()
  add_modular_library(LLCLTelemetry
    STATIC OBJECT # Always generate an obj.${name} target.

    TelemetryContext.cpp
    FileMetricExporter.cpp
    FileLogExporter.cpp
    )
endif()
