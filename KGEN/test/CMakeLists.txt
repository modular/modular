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

# This is the top-level test target for all things Mojo. Subdirectories each
# define their own test targets, and add those targets as dependencies of this
# one.
add_custom_target(check-mojo)
set_target_properties(check-mojo PROPERTIES FOLDER "Tests")
add_dependencies(check-all check-mojo)

add_custom_command(
  OUTPUT mojo_warmup_parser_cache.mlir mojo_warmup_parser_cache.input.mlir
  DEPENDS kgen-translate
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND echo "" > mojo_warmup_parser_cache.input.mlir
  COMMAND $<TARGET_FILE:kgen-translate> -import-mojo
                                        -o mojo_warmup_parser_cache.input.mlir
)
add_custom_target(warmup-mojo-parser-cache DEPENDS mojo_warmup_parser_cache.mlir)

add_subdirectory(kgen)

add_subdirectory(mojo-parser)
add_subdirectory(mojo-isolated)

add_subdirectory(mojo-tool)
add_subdirectory(mojo-integration)

add_subdirectory(mojo-debug)
add_subdirectory(mojo-jupyter)
add_subdirectory(mojo-lsp-server)
add_subdirectory(mojo-repl)
add_subdirectory(mojo-test-executor)
add_subdirectory(mojo-mogg)
add_subdirectory(test-packages)
