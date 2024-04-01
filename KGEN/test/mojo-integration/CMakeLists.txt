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

add_modular_testsuite(check-mojo-integration
  TOP_LEVEL_CHECK check-mojo

  TOOLS
  mojo-stdlib
  kgen
  kgen-translate
  kgen-opt
  llvm-module-split
)
