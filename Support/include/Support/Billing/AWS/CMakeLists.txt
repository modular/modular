##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# TODO: Move MAWSUtils here
add_modular_library(MAWSBilling
  AWSBillingContext.cpp
  InstanceIdentifier.cpp

  LINK_LIBS PUBLIC
  MBilling
  MSupport
  MSupportHTTP
  MThreading
  libcurl
)
