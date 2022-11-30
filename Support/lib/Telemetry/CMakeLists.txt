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
# setting the include_directories as `SYSTEM`.  For now, just explicitly set the
# underlying `SYSTEM_INTERFACE_INCLUDE_DIRECTORIES` to silence warnings and
# treat OpenTelemetry properly as a third-party library.
function(target_link_libraries_system target)
  set(libs ${ARGN})
  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target} SYSTEM PUBLIC ${lib_include_dirs})
    target_link_libraries(${target} ${lib})
  endforeach(lib)
endfunction()

target_link_libraries_system(MSupportTelemetry
  opentelemetry_exporter_ostream_metrics
  opentelemetry_metrics
  opentelemetry_resources
)
