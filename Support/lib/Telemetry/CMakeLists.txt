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

  add_modular_library(MTelemetry
    TelemetryContext.cpp

    LINK_LIBS PUBLIC
    opentelemetry_metrics

    # It's important to keep OTLP PRIVATE. If not, it will leak our protobuf headers
    # across the codebase given that LLCL links with MTelemetry, and stock TF headers
    # will throw a build-time error because they were generated with an older protobuf.
    LINK_LIBS PRIVATE
    opentelemetry_exporter_otlp_http
    opentelemetry_exporter_otlp_http_metric
    opentelemetry_resources
    )

  target_compile_definitions(MTelemetry PUBLIC MODULAR_ENABLE_TELEMETRY=${MODULAR_ENABLE_TELEMETRY})

else()
  add_modular_library(MTelemetry
    TelemetryContext.cpp
    )
endif()
