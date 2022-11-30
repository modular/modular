##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddOpenTelemetry)

add_modular_library(MSupportTelemetry
  Context.cpp
)

target_link_libraries(MSupportTelemetry
  PUBLIC
  opentelemetry_exporter_ostream_metrics
  opentelemetry_metrics
  opentelemetry_resources
)
