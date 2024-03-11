##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

add_kgen_kernel(mojo-build-server-static
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/mojo_build_server.mojo
)

add_modular_tool(mojo-build-server
  empty.c

  LINK_LIBS PRIVATE
  mojo-build-server-static
  KGENCompilerRT-static
)
