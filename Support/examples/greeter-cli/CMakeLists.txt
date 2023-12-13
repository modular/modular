##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# This adds CMake targets to:
# 1. Compile and link the `greeter-cli` executable from C++ source code.
# 2. Use TableGen to generate C++ and documentation for that executable.
#
##===----------------------------------------------------------------------===##

# The `greeter-cli` executable is made up of the main source file, plus one
# source file for each subcommand.
add_modular_tool(greeter-cli
  greeter-cli.cpp
  Bye/bye.cpp
  Hi/hi.cpp

  LINK_LIBS PRIVATE
  # LLVMOption implements command line argument parsing helpers.
  LLVMOption
  # MSupportDriver implements Modular CLI helpers, like "print an error if we
  # don't recognize one of the command line arguments."
  MSupportDriver
)
# We'll be generating files using TableGen that we wish to include in our CLI
# source code, so we add the generated file location to our include paths.
target_include_directories(greeter-cli PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

# WARNING: Because `greeter-cli` is just an example program, we prevent it from
# being built in the `build/bin` directory. If you're adding a CLI executable,
# you probably DO want it placed there, and so you should remove this call to
# `set_output_directory`!
set_output_directory(greeter-cli
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# The rest of this file sets up the TableGen targets that generate:
# - `llvm-tblgen -gen-opt-parser-defs`: the C++ used by LLVMOption to parse
#   options.
# - `driver-tblgen`: the documentation for `greeter-cli`, such as its help text
#   string and Markdown manual.
#
# First, we define an umbrella target that builds all of our TableGen targets.
add_custom_target(GreeterCLIDriverOptions)
# Then, we allow our subcommands to add their TableGen targets.
add_subdirectory(Bye)
add_subdirectory(Hi)
# And then create our own TableGen targets.
add_driver_option_tablegen(GreeterCLI DriverOptions.td
  # MANUAL_NAME: The file name for the Linux manual page file.
  MANUAL_NAME greeter-cli
  # MARKDOWN_NAME: The file name for the Markdown file. Top-level commands
  # often use "index" because static site generators look for an "index.md"
  # file.
  MARKDOWN_NAME index
  # EXTRA_INCLUDES: Extra `-I` includes to pass to TableGen invocations.
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})
# Finally, we create CMake targets that copy the generated Markdown files to
# a single directory.
collect_driver_markdown(GreeterCLIDriverOptions)

# Add a dependency, so building the CLI also generates all its TableGen.
add_dependencies(greeter-cli GreeterCLIDriverOptions)
