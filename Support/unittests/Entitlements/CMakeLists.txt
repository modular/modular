##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# For the tests, we only allow the local root certificate and key.
file(READ ${CMAKE_SOURCE_DIR}/Support/lib/Entitlements/RootCertificates/local/cert.pem CERTIFICATE_PEM)
file(READ ${CMAKE_SOURCE_DIR}/Support/lib/Entitlements/RootCertificates/local/key.pem KEY_PEM)
file(READ ${CMAKE_SOURCE_DIR}/Support/lib/Entitlements/RootCertificates/local/endpoint.txt AUTH_URL)

configure_file(${CMAKE_SOURCE_DIR}/Support/lib/Entitlements/RootCertificates/RootCert.inc.in ${CMAKE_CURRENT_BINARY_DIR}/RootCertificates/RootCert.inc @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/RootCertificates)

add_modular_unittest(
  TOP_LEVEL_CHECK
  check-support

  SUITE_NAME
  SupportEntitlementUnitTests

  TEST_NAME
  SupportEntitlementTests

  EntitlementsTest.cpp
  EntitlementTokenTest.cpp

  LINK_LIBS PRIVATE
  MSupportEntitlements
)
