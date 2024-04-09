##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##


if(NOT TARGET max-pybind-ext)
  return()
endif()

set(CUSTOM_OP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../Inputs/test_user_op)
set(CUSTOM_OP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/user_op.mojopkg)

add_custom_command(
  COMMAND $<TARGET_FILE:mojo> package ${CUSTOM_OP_SOURCE} -o ${CUSTOM_OP_OUTPUT}
  OUTPUT ${CUSTOM_OP_OUTPUT}
  DEPENDS
    ${CUSTOM_OP_SOURCE}/__init__.mojo
    ${CUSTOM_OP_SOURCE}/user_add.mojo
    ${CUSTOM_OP_SOURCE}/user_ops.mojo
    ${CUSTOM_OP_SOURCE}/user_sqrt.mojo
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

  TAGS
  gpu
)
