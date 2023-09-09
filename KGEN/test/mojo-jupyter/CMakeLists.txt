##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

add_modular_testsuite(check-mojo-jupyter
  TOP_LEVEL_CHECK check-mojo
  TOOLS
  mojo-sdk
)

if(LLDB_FOUND)
  add_dependencies(check-mojo-jupyter MojoLLDB mojo-jupyter-executor)
endif()
