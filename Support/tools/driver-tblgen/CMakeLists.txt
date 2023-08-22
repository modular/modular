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
  GenManPage.cpp
  GenMarkdown.cpp)

# Set this on the global scope so that we don't have to pipe it up the stack
# manually.
set(DRIVER_TABLEGEN_EXE $<TARGET_FILE:driver-tblgen>
    CACHE INTERNAL "DRIVER_TABLEGEN_EXE")

# Given a prefix "Foo" and an input path "Bar.td", this creates several public
# TableGen targets:
#
# 1. "FooDriverBar", which when built outputs option parser definitions to a
#    file "Bar.inc".
# 2. "FooBarHelpText", which when built outputs a raw C++ string literal
#    containing help text to a file "BarHelpText.inc".
# 3. "FooBarManPage", which when built generates a manual page file. The name of
#    the manual page is determined by the `MANUAL_NAME` argument, or "Bar" by
#    default (with '1' appended, since '1' is the "general commands" section for
#    manual pages). That file is also copied to the build's share/man directory.
# 4. "FooBarMarkdown", which when built generates a Markdown file. The name of
#    the Markdown file is determined by the `MARKDOWN_FILE` argument, or "Bar"
#    by default (with ".md" appended). The path of the generated file is also
#    added to a list property named "DRIVER_MARKDOWN_PATHS" that is associated
#    with the target "FooDriverOptions".
#
# Each of these TableGen targets is added to the driver option umbrella target
# "FooDriverOptions".
macro(add_driver_option_tablegen prefix input)
  cmake_parse_arguments(ARG
    ""
    "MANUAL_NAME;MARKDOWN_NAME"
    "EXTRA_INCLUDES"
    ${ARGN})
  set(tblgen_includes)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out empty items before prepending each entry with -I
  list(REMOVE_ITEM tblgen_includes "")
  list(TRANSFORM tblgen_includes PREPEND -I)

  get_filename_component(name "${input}" NAME_WE)
  # The name of the umbrella target that depends upon all targets created by
  # this macro.
  set(target ${prefix}DriverOptions)

  # 1. Option parser definitions
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(LLVM "${name}.inc" -gen-opt-parser-defs
    EXTRA_INCLUDES ${ARG_EXTRA_INCLUDES})
  add_public_tablegen_target("${prefix}Driver${name}")
  add_dependencies(${target} "${prefix}Driver${name}")

  # 2. Help text
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(DRIVER "${name}HelpText.inc" -gen-help-text
    EXTRA_INCLUDES ${ARG_EXTRA_INCLUDES})
  add_public_tablegen_target("${prefix}${name}HelpText")
  add_dependencies(${target} "${prefix}${name}HelpText")

  # 3. Manual page: the 'GenerateManPage' target generates the man page file
  # itself, and the 'ManPage' target copies it to the 'share' directory.
  if (ARG_MANUAL_NAME)
    set(manual_name "${ARG_MANUAL_NAME}.1")
  else()
    string(TOLOWER "${name}.1" manual_name)
  endif()
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(DRIVER "${manual_name}" -gen-man-page
    EXTRA_INCLUDES ${ARG_EXTRA_INCLUDES})
  add_public_tablegen_target("${prefix}${name}GenerateManPage")

  add_custom_command(OUTPUT "${MODULAR_SHARE_DIR}/man/man1/${manual_name}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MODULAR_SHARE_DIR}/man/man1"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_BINARY_DIR}/${manual_name}"
      "${MODULAR_SHARE_DIR}/man/man1/${manual_name}"
    DEPENDS "${prefix}${name}GenerateManPage")
  add_custom_target("${prefix}${name}ManPage"
    DEPENDS "${MODULAR_SHARE_DIR}/man/man1/${manual_name}")

  add_dependencies(${target} "${prefix}${name}ManPage")

  # 4. Markdown
  if (ARG_MARKDOWN_NAME)
    set(markdown_name "${ARG_MARKDOWN_NAME}.md")
  else()
    set(markdown_name "${name}.md")
  endif()
  set(LLVM_TARGET_DEFINITIONS "${input}")
  tablegen(DRIVER "${markdown_name}" -gen-markdown
    EXTRA_INCLUDES ${ARG_EXTRA_INCLUDES})
  add_public_tablegen_target("${prefix}${name}Markdown")

  set(markdown_path "${CMAKE_CURRENT_BINARY_DIR}/${markdown_name}")
  get_property(is_paths_set TARGET ${target} PROPERTY DRIVER_MARKDOWN_PATHS SET)
  if (is_paths_set)
    get_target_property(paths ${target} DRIVER_MARKDOWN_PATHS)
    set_target_properties(${target}
      PROPERTIES DRIVER_MARKDOWN_PATHS "${paths};${markdown_path}")
  else()
    set_target_properties(${target}
      PROPERTIES DRIVER_MARKDOWN_PATHS "${markdown_path}")
  endif()

  add_dependencies(${target} "${prefix}${name}Markdown")
endmacro()

# Given a target name "BarTarget", adds a custom command target that when built
# copies all items in list property named "DRIVER_MARKDOWN_PATHS" that's
# associated with that target. This list property is defined by the
# `add_driver_option_tablegen` macro.
macro(collect_driver_markdown target)
  get_property(markdown_paths TARGET ${target} PROPERTY DRIVER_MARKDOWN_PATHS)
  foreach(path ${markdown_paths})
    add_custom_command(TARGET ${target}
      COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/markdown"
      COMMAND ${CMAKE_COMMAND} -E copy
        ${path} "${CMAKE_CURRENT_BINARY_DIR}/markdown"
      DEPENDS ${path})
  endforeach()
endmacro()
