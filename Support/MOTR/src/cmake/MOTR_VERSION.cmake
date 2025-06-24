##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Parse the version.txt file and set the MOTR_VERSION_MAJOR, MOTR_VERSION_MINOR,
# MOTR_VERSION_PATCH, and MOTR_VERSION_STRING variables in the parent scope
function(parse_version_file)
    if(EXISTS "${CMAKE_SOURCE_DIR}/../../version.txt")
        file(READ "${CMAKE_SOURCE_DIR}/../../version.txt" VERSION_CONTENT)
        string(STRIP "${VERSION_CONTENT}" VERSION_CONTENT)
        string(REPLACE "." ";" VERSION_PARTS "${VERSION_CONTENT}")
        list(GET VERSION_PARTS 0 MOTR_VERSION_MAJOR)
        list(GET VERSION_PARTS 1 MOTR_VERSION_MINOR)
        list(GET VERSION_PARTS 2 MOTR_VERSION_PATCH)
        set(MOTR_VERSION_MAJOR "${MOTR_VERSION_MAJOR}" PARENT_SCOPE)
        set(MOTR_VERSION_MINOR "${MOTR_VERSION_MINOR}" PARENT_SCOPE)
        set(MOTR_VERSION_PATCH "${MOTR_VERSION_PATCH}" PARENT_SCOPE)
        set(MOTR_VERSION_STRING "${MOTR_VERSION_MAJOR}.${MOTR_VERSION_MINOR}.${MOTR_VERSION_PATCH}" PARENT_SCOPE)

        # Add configure_file to track version changes
        configure_file(
            "${CMAKE_SOURCE_DIR}/../../version.txt"
            "${CMAKE_BINARY_DIR}/version.txt.timestamp"
            COPYONLY
        )
    else()
        message(FATAL_ERROR "version.txt not found at ${CMAKE_SOURCE_DIR}/../../version.txt")
    endif()
endfunction()

parse_version_file()

message(STATUS "MOTR_VERSION_MAJOR: ${MOTR_VERSION_MAJOR}")
message(STATUS "MOTR_VERSION_MINOR: ${MOTR_VERSION_MINOR}")
message(STATUS "MOTR_VERSION_PATCH: ${MOTR_VERSION_PATCH}")
message(STATUS "MOTR_VERSION_STRING: ${MOTR_VERSION_STRING}")

set(GIT_COMMIT "0000000000000000000000000000000000000000")
execute_process(COMMAND git rev-parse HEAD OUTPUT_VARIABLE GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)

function(set_motr_version_defines TARGET)
    target_compile_definitions(${TARGET} PUBLIC
        MOTR_VERSION_MAJOR=${MOTR_VERSION_MAJOR}
        MOTR_VERSION_MINOR=${MOTR_VERSION_MINOR}
        MOTR_VERSION_PATCH=${MOTR_VERSION_PATCH}
        MOTR_VERSION_STRING="${MOTR_VERSION_STRING}"
        GIT_COMMIT="${GIT_COMMIT}"
    )
endfunction()
