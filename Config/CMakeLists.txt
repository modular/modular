##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# Generate the Modular build config files.
#
##===----------------------------------------------------------------------===##

##===----------------------------------------------------------------------===##
# Version
##===----------------------------------------------------------------------===##

find_first_existing_vc_file("${MODULAR_MAIN_SRC_DIR}" modular_vc)
set(version_inc "${MODULAR_BINARY_DIR}/Config/include/Config/Version.h.inc")
set(generate_version_script "${MODULAR_SOURCE_DIR}/Config/GenerateVersion.cmake")

# Create custom target to generate the version include, so that it regenerates
# properly.
add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${modular_vc}" "${generate_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DMODULAR_SOURCE_DIR=${MODULAR_SOURCE_DIR}"
                           "-DMODULAR_BINARY_DIR=${MODULAR_BINARY_DIR}"
                           "-DMODULAR_VERSION_MAJOR=${MODULAR_VERSION_MAJOR}"
                           "-DMODULAR_VERSION_MINOR=${MODULAR_VERSION_MINOR}"
                           "-DMODULAR_VERSION_PATCH=${MODULAR_VERSION_PATCH}"
                           -P "${generate_version_script}")

# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

##===----------------------------------------------------------------------===##
# Config
##===----------------------------------------------------------------------===##

set(config_inc "${MODULAR_BINARY_DIR}/Config/include/Config/Config.h")

configure_file(
  ${MODULAR_SOURCE_DIR}/Config/Config.h.in
  ${config_inc}
)

add_custom_target(MConfig ALL DEPENDS
  "${version_inc}"
  "${config_inc}"
)
set_target_properties(MConfig PROPERTIES FOLDER "Misc")
