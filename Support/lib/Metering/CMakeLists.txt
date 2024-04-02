##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MMetering
  MeteringContext.cpp

  AWS/InstanceIdentifier.cpp

  LINK_LIBS PUBLIC
  MSupport
  MSupportHTTP
  MTelemetry
  MThreading
  libcurl
)
