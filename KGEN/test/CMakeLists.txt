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

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_modular_testsuite(check-mojo
  TOOLS
  llvm-objdump
  kgen
  kgen-opt
  kgen-translate
  mojo-sdk
)

add_subdirectory(kgen)
add_subdirectory(mojo-integration)
add_subdirectory(mojo-jupyter)
add_subdirectory(mojo-lldb-dwarf)
add_subdirectory(mojo-lldb-repl)
add_subdirectory(mojo-lsp-server)
add_subdirectory(mojo-parser)
