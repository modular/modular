##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# Configure the Modular testsuite.
#
##===----------------------------------------------------------------------===##

include(AddPython)
modular_find_python3_development()

add_subdirectory(pytests)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

add_modular_testsuite(check-kgen
  TOOLS
  llvm-objdump
  kgen
  kgen-opt
  kgen-translate
  mojo-lsp-server
  mojo-sdk
)

if(LLDB_FOUND)
  add_dependencies(check-kgen MojoLLDB mojo-jupyter-executor)
endif()
