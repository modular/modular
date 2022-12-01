##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddOpenTelemetry)

if(${DISABLE_ALL_EXTERNAL})
  return()
endif()

add_modular_library(MSupportTelemetry
  Context.cpp
)

# The `SYSTEM_INTERFACE_INCLUDE_PROPERTIES` does not seem to get respected for
# these import targets in OpenTelemetry when used by UseRepository, despite us
# setting the include_directories as `SYSTEM`.
target_link_libraries_system(MSupportTelemetry PUBLIC
  opentelemetry_exporter_ostream_metrics
  opentelemetry_metrics
  opentelemetry_resources
)
