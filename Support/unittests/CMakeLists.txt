##===- CMakeLists.txt -----------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_modular_unittest(
  SUITE_NAME
  SupportUnitTests

  TEST_NAME
  SupportTests

  HMACTest.cpp
  BlobCacheTest.cpp

  LINK_LIBS
  PRIVATE
  MSupport)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
