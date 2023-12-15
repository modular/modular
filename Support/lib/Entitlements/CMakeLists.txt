##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(AddMbedTLS)

# TODO: Figure out a way to make this safer - we do NOT want dev certs in production.
if(NOT MODULAR_ROOT_CERTIFICATE)
  set(MODULAR_ROOT_CERTIFICATE "dev" CACHE STRING "The root certificate to use. Options are {dev, staging, prod}. `dev` certificates MUST NOT be used in production.")
endif()

if (${MODULAR_ROOT_CERTIFICATE} STREQUAL "dev")
  # This certificate and key were generated with:
  # ```shell
  # openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -keyout Support/lib/Entitlements/RootCertificates/dev/key.pem -out Support/lib/Entitlements/RootCertificates/dev/cert.pem -sha256 -days 3650 -nodes -subj "/C=US/O=Modular Inc/CN=dev.auth.modular.com" -addext "keyUsage=critical,cRLSign,digitalSignature,keyCertSign" -addext "basicConstraints=critical,CA:TRUE,pathlen:0" -addext "authorityKeyIdentifier=keyid:always,issuer:always" -addext "subjectKeyIdentifier=hash"
  # openssl ec -in Support/lib/Entitlements/RootCertificates/dev/key.pem -out Support/lib/Entitlements/RootCertificates/dev/key.pem
  # ```
  # It's a self-signed certificate that we can use as the root of the client certificate chains.
  file(READ RootCertificates/dev/cert.pem CERTIFICATE_PEM)
  file(READ RootCertificates/dev/key.pem KEY_PEM)
  message(STATUS "DO NOT SHIP THIS BUILD! You are using the dev certificate for the modular entitlements library. DO NOT SHIP THIS BUILD!")
elseif (${MODULAR_ROOT_CERTIFICATE} STREQUAL "staging")
  file(READ RootCertificates/staging/cert.pem CERTIFICATE_PEM)
  message(STATUS "DO NOT SHIP THIS BUILD! You are using the staging certificate for the modular entitlements library. DO NOT SHIP THIS BUILD!")
else ()
  message(FATAL_ERROR "Unknown value for MODULAR_ROOT_CERTIFICATE: ${MODULAR_ROOT_CERTIFICATE}, expected one of {dev, staging, prod}")
endif ()

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
