##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

if(MODULAR_ENABLE_TELEMETRY)
  include(AddOpenTelemetry)

  add_flag_if_supported("-Werror -Wno-covered-switch-default" SUPPORTS_TELEMETRYWNO_COVERED_SWITCH_DEFAULT)

  add_modular_library(MTelemetry
    TelemetryContext.cpp
    FileMetricExporter.cpp
    FileLogExporter.cpp

    LINK_LIBS PUBLIC
    MSupport
    MSupportSettings
    MArchTarget
    MVersion
    opentelemetry_metrics
    opentelemetry_logs

    # It's important to keep OTLP PRIVATE. If not, it can leak our protobuf headers
    # across the codebase when linking with MTelemetry, and stock TF headers will
    # throw a build-time error because they were generated with an older protobuf.
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
  add_modular_library(MTelemetry
    TelemetryContext.cpp
    FileMetricExporter.cpp
    FileLogExporter.cpp

    LINK_LIBS PUBLIC
    MSupportSettings
    )
endif()
