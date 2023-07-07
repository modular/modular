##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

add_tablegen(driver-tblgen DRIVER
  driver-tblgen.cpp
  DriverCommand.cpp
  GenHelpText.cpp
  GenManPage.cpp)

# Set this on the global scope so that we don't have to pipe it up the stack manually.
set(DRIVER_TABLEGEN_EXE $<TARGET_FILE:driver-tblgen> CACHE INTERNAL "DRIVER_TABLEGEN_EXE")

# Given a prefix "Foo", an input path "Bar.td" and a manual page name "bar",
# this creates 3 public TableGen targets:
#
# 1. "FooDriverBar", which when built outputs option parser definitions to a
#    file "Bar.inc".
# 2. "FooBarHelpText", which when built outputs a raw C++ string literal
#    containing help text to a file "BarHelpText.inc".
# 3. "FooBarManPage", which when built generates a manual page file named
#    "bar.1" ('1' is the "general commands" section for manual pages). That
#    "bar.1" file is also copied to the build's share/man directory.
#
# Each of these TableGen targets is added to the driver option umbrella target.
macro(add_driver_option_tablegen prefix input manual_name)
  get_filename_component(name "${input}" NAME_WE)

  # 1. Option parser definitions
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(LLVM "${name}.inc" -gen-opt-parser-defs)
  add_public_tablegen_target("${prefix}Driver${name}")
  add_dependencies(${prefix}DriverOptions "${prefix}Driver${name}")

  # 2. Help text
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(DRIVER "${name}HelpText.inc" -gen-help-text)
  add_public_tablegen_target("${prefix}${name}HelpText")
  add_dependencies(${prefix}DriverOptions "${prefix}${name}HelpText")

  # 3. Manual page
  set(file_name "${manual_name}.1")
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(DRIVER "${file_name}" -gen-man-page)
  add_public_tablegen_target("${prefix}${name}ManPage")

  add_custom_command(TARGET "${prefix}${name}ManPage" POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MODULAR_SHARE_DIR}/man/man1"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_CURRENT_BINARY_DIR}/${file_name}"
      "${MODULAR_SHARE_DIR}/man/man1/${file_name}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${file_name}")

  add_dependencies(${prefix}DriverOptions "${prefix}${name}ManPage")
endmacro()
