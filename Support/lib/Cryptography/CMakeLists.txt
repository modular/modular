##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddMbedTLS)

add_modular_library(MSupportCryptography
  Keypair.cpp
  Symmetric.cpp

  LINK_LIBS PUBLIC
  MSupport
  MbedTLS::mbedcrypto
)
