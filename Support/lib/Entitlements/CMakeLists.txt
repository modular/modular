##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddMbedTLS)

add_modular_library(MSupportEntitlements
  Entitlement.cpp
  EntitlementStore.cpp

  LINK_LIBS PUBLIC
  MSupport
  MSupportASN1
  MSupportCryptography
  MbedTLS::mbedcrypto
  MbedTLS::mbedx509
)

if (APPLE)
  find_library(CF CoreFoundation)
  if (NOT CF)
    message(FATAL_ERROR "could not find CoreFoundation framework")
  endif()

  find_library(SEC Security)
  if (NOT SEC)
    message(FATAL_ERROR "could not find Security framework")
  endif()

  target_link_libraries(MSupportEntitlements PUBLIC ${CF} ${SEC})
endif()
