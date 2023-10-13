##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_modular_testsuite(check-api-integration-tests
  TOOLS
  c-api-validator
  modular-api-executor
  mojo

  DEPENDS
  tf
  tfl
  monnx
  mof
  modular-framework-common
  modular-framework-tf-ext
  )

add_subdirectory(python)
