##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

find_first_existing_vc_file("${MODULAR_MAIN_SRC_DIR}" modular_vc)
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/GeneratedVersion.h")
set(generate_version_script "${CMAKE_CURRENT_SOURCE_DIR}/GenerateVersion.cmake")

# Create custom target to generate the version include, so that it regenerates
# properly.
add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${modular_vc}" "${generate_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DMODULAR_SOURCE_DIR=${MODULAR_SOURCE_DIR}"
                           "-DVERSION_OUTPUT_FILE=${version_inc}"
                           "-DMODULAR_VERSION_MAJOR=${MODULAR_VERSION_MAJOR}"
                           "-DMODULAR_VERSION_MINOR=${MODULAR_VERSION_MINOR}"
                           "-DMODULAR_VERSION_PATCH=${MODULAR_VERSION_PATCH}"
                           "-DMODULAR_VERSION_LABEL=${MODULAR_VERSION_LABEL}"
                           "-DMODULAR_DERIVED_DIR=${MODULAR_DERIVED_DIR}"
                           "-DMODULAR_BUILD_TYPE_LOWER=${MODULAR_BUILD_TYPE_LOWER}"
  -P "${generate_version_script}")

# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

add_custom_target(MVersionGeneratedHeader DEPENDS "${version_inc}")

add_modular_library(MVersion
  STATIC OBJECT # Always generate an obj.${name} target.

  Version.cpp

  DEPENDS MVersionGeneratedHeader
)
target_include_directories(MVersion PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

set_target_properties(MVersion PROPERTIES FOLDER Misc)
set_target_properties(MVersionGeneratedHeader PROPERTIES FOLDER Misc)
