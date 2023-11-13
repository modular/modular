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

add_subdirectory(kgen)
add_subdirectory(mojo-debug)
add_subdirectory(mojo-integration)
add_subdirectory(mojo-jupyter)
add_subdirectory(mojo-lsp-server)
add_subdirectory(mojo-parser)
add_subdirectory(mojo-repl)
add_subdirectory(mojo-tool)
