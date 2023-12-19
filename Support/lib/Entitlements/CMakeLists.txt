##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddMbedTLS)

# TODO: Figure out a way to make this safer - we do NOT want dev certs in production.
if(NOT MODULAR_ROOT_CERTIFICATE)
  set(MODULAR_ROOT_CERTIFICATE "local" CACHE STRING "The root certificate to use. Options are {local, dev, staging, prod}. Certificates other than `prod` MUST NOT be used in production.")
endif()

set(accepted_root_cert_kinds "local;dev;staging;prod")
if (NOT ${MODULAR_ROOT_CERTIFICATE} IN_LIST accepted_root_cert_kinds)
  message(FATAL_ERROR "Unknown value for MODULAR_ROOT_CERTIFICATE: ${MODULAR_ROOT_CERTIFICATE}, expected one of {local, dev, staging, prod}")
endif()

file(READ RootCertificates/${MODULAR_ROOT_CERTIFICATE}/cert.pem CERTIFICATE_PEM)
file(READ RootCertificates/${MODULAR_ROOT_CERTIFICATE}/key.pem KEY_PEM)
file(READ RootCertificates/${MODULAR_ROOT_CERTIFICATE}/endpoint.txt AUTH_URL)
if (NOT (${MODULAR_ROOT_CERTIFICATE} STREQUAL "prod"))
  message(STATUS "DO NOT SHIP THIS BUILD! You are using the ${MODULAR_ROOT_CERTIFICATE} certificate for the modular entitlements library. DO NOT SHIP THIS BUILD!")
endif()

# Include the root cert here.
configure_file(RootCertificates/RootCert.inc.in ${CMAKE_CURRENT_BINARY_DIR}/RootCertificates/RootCert.inc @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/RootCertificates)

add_modular_library(MSupportEntitlements
  Entitlement.cpp
  EntitlementStore.cpp

  LINK_LIBS PUBLIC
  MSupport
  MSupportASN1
  MSupportCryptography
  MSupportHTTP
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
