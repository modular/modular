##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# Configure the SDK integration testsuite.
#
##===----------------------------------------------------------------------===##

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_modular_testsuite(check-sdk-integration
  TOOLS
  FileCheck
  max

  DEPENDS
  mtorch

  TAGS
  asan-red
)
