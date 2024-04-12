##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(TORCH_LIB "missing")
set(FRAMEWORK_DYLIB_DEPS "")
set(FRAMEWORK_TEST_DEPS "")
if(TARGET mtorch)
  set(TORCH_LIB "mtorch")
  list(APPEND FRAMEWORK_DYLIB_DEPS mtorch)
  list(APPEND FRAMEWORK_TEST_DEPS pytorch-generated-tests)
endif()

if(TARGET tf)
  set(TF_LIB "tf")
  list(APPEND FRAMEWORK_DYLIB_DEPS tf)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/framework-lib-path.cfg
  CONTENT $<TARGET_FILE:modular-framework-common>
)

add_modular_testsuite(check-max-engine-integration-tests
  TOOLS
  c-api-validator
  modular-api-executor
  mojo
  mt

  DEPENDS
  monnx
  mof
  ${FRAMEWORK_DYLIB_DEPS}
  ${FRAMEWORK_TEST_DEPS}
  is-cuda-available
  max-mojo
  kernels-register

  TAGS
  gpu
  asan-red
)

add_subdirectory(python)
