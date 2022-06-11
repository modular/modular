#===- CMakeLists.txt ------------------------------------------*- cmake -*-===##
#
# This file is Modular Inc proprietary.
#
#===-----------------------------------------------------------------------===##
#
# Configure the Modular testsuite.
#
#===-----------------------------------------------------------------------===##

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(KGEN_TEST_DEPENDS
  FileCheck count not
  kgen-opt
  )

add_lit_testsuite(check-kgen "Running the KGEN regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS ${KGEN_TEST_DEPENDS}
  )
set_target_properties(check-kgen PROPERTIES FOLDER "Tests")

add_lit_testsuites(KGEN ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${KGEN_TEST_DEPENDS}
)

add_dependencies(check-all check-kgen)
