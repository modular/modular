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

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/imported_module.mojo
  ${CMAKE_CURRENT_BINARY_DIR}/Inputs/imported_module.mojo
  COPYONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/imported_module_2.mojo
  ${CMAKE_CURRENT_BINARY_DIR}/Inputs/imported_module_2.mojo
  COPYONLY
)

add_modular_testsuite(check-mojo-repl
  TOP_LEVEL_CHECK check-mojo
  TOOLS
  mojo-sdk
  mojo
)
