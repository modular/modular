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
endif()
