##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddMbedTLS)

add_modular_library(MSupportCryptography
  Keypair.cpp

  LINK_LIBS PUBLIC
  MSupport
  MbedTLS::mbedcrypto
)
target_include_directories(MSupportCryptography
  PUBLIC
  "${MBEDTLS_INCLUDE_DIRS}"
)
