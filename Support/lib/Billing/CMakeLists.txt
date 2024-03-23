##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_library(MBilling
  BillingContext.cpp

  AWS/InstanceIdentifier.cpp

  LINK_LIBS PUBLIC
  MSupport
  MSupportHTTP
  MThreading
  libcurl
)
