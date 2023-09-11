##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddCurl)

add_modular_library(MSupportHTTP
  HTTPClient.cpp

  LINK_LIBS PUBLIC
  libcurl
  MSupport
)
