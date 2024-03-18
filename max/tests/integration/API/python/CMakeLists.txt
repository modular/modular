##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


if(NOT TARGET max-pybind-ext)
  return()
endif()

set(CUSTOM_OP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/user_op.mojopkg)

add_custom_command(
  COMMAND $<TARGET_FILE:mojo> package ${CMAKE_CURRENT_SOURCE_DIR}/../Inputs/test_user_op -o ${CUSTOM_OP_OUTPUT}
  OUTPUT ${CUSTOM_OP_OUTPUT}
)

add_modular_pytest(max-engine

  TOP_LEVEL_CHECK
  check-sdk

  EXTRA_ARGS
  --custom-ops-path=${CUSTOM_OP_OUTPUT}

  EXTRA_DIRS_FOR_ASAN
  $<TARGET_FILE_DIR:mof>

  DEPENDS
  ${CUSTOM_OP_OUTPUT}
  mof
  is-cuda-available
  max-pybind-ext
)
