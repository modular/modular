##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# Building for desktop (WebGPU-native) with Dawn:
#  1. git clone https://github.com/google/dawn dawn
#  2. cmake -B build -DIMGUI_DAWN_DIR=dawn
#  3. cmake --build build
# The resulting binary will be found at one of the following locations:
#   * build/Debug/example_glfw_wgpu[.exe]
#   * build/example_glfw_wgpu[.exe]

# Building for Emscripten:
#  1. Install Emscripten SDK following the instructions: https://emscripten.org/docs/getting_started/downloads.html
#  2. Install Ninja build system
#  3. emcmake cmake -G Ninja -B build
#  3. cmake --build build
#  4. emrun build/index.html

cmake_minimum_required(VERSION 3.11)
project(motr-gui C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)

# Get absolute path to the MOTR third-party directory
get_filename_component(THIRD_PARTY_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third-party/MOTR"
  ABSOLUTE
)
message(STATUS "Third party directory: ${THIRD_PARTY_DIR}")


include(FetchContent)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)

FetchContent_Declare(
  yoga
  GIT_REPOSITORY https://github.com/facebook/yoga.git
  GIT_TAG 042f5013152eb81c1552dec945b88f7b95ca350f #refs/tags/v3.2.1
)

FetchContent_Declare(
  tweeny
  GIT_REPOSITORY https://github.com/mobius3/tweeny.git
  GIT_TAG 3f7312034add5be888360e62005f596294ac93b0 # v3.2.0
)

FetchContent_MakeAvailable(glm)
FetchContent_MakeAvailable(yoga)
FetchContent_MakeAvailable(tweeny)

function(add_to_var varname)
  string(JOIN " " args_with_spaces ${ARGN})
  set(tmp "${${varname}} ${args_with_spaces}")
  set("${varname}" "${tmp}" PARENT_SCOPE)
  message(STATUS "${varname}=${tmp}")
endfunction()


# Path substitutions
# /emsdk/emscripten -> emscripten
# /opt/homebrew/Cellar/emscripten/4.0.8/libexec -> emscripten
# ${MODULAR_ROOT}/Support/MOTR/build/gui/web/Debug/ -> ""
# ${MODULAR_ROOT}/Support/MOTR/ -> ""

# Libraries
if(EMSCRIPTEN)
  # https://github.com/clangd/clangd/issues/1621#issuecomment-1703330511
  # explicitly expose the e++ compiler flags for compile_commands.json -> intellisense
  execute_process(COMMAND em++ --cflags OUTPUT_VARIABLE EM_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "EM_CFLAGS=${EM_CFLAGS}")

  # Debug enable source maps and separate DWARF debug info
  add_to_var(CMAKE_CXX_FLAGS_DEBUG
    -O0
    -g3
    -fno-inline
    -gsplit-dwarf
    -gdwarf-5
  )


  # Link-time flags for Debug
  # Note that there is a chrome bug
  # where source maps are not generated
  # when -gseparate-dwarf is used
  # https://issues.chromium.org/issues/396169337
  add_to_var(CMAKE_EXE_LINKER_FLAGS_DEBUG
    -O0
    -g3
    # -sSAFE_HEAP=1
    -sASSERTIONS=1
    -sDISABLE_EXCEPTION_CATCHING=0
    -sINITIAL_MEMORY=1024MB
    -sWASM_BIGINT=1
    -sERROR_ON_WASM_CHANGES_AFTER_LINK=1
    # -sALLOW_MEMORY_GROWTH=1
    -fno-inline
    -gdwarf-5
    -gsplit-dwarf
    -sREVERSE_DEPS=all
  )


  if(EMSCRIPTEN_VERSION VERSION_GREATER_EQUAL "3.1.57")
    set(IMGUI_EMSCRIPTEN_GLFW3 "--use-port=contrib.glfw3" CACHE STRING "Choose between --use-port=contrib.glfw3 and -sUSE_GLFW=3 for GLFW implementation (default to --use-port=contrib.glfw3)")
  else()
    # cannot use contrib.glfw3 prior to 3.1.57
    set(IMGUI_EMSCRIPTEN_GLFW3 "-sUSE_GLFW=3" CACHE STRING "Use -sUSE_GLFW=3 for GLFW implementation" FORCE)
  endif()
  set(LIBRARIES glfw)
  add_compile_options(
    -sDISABLE_EXCEPTION_CATCHING=1
    # -sNO_DISABLE_EXCEPTION_THROWING
    # -sEXCEPTION_STACK_TRACES=1
    # -sASSERTIONS=1
    # -DIMGUI_DISABLE_FILE_FUNCTIONS=1
  )
  add_compile_options(-fno-exceptions)
  add_compile_options(-fno-rtti)
else()
  # Dawn wgpu desktop
  set(DAWN_FETCH_DEPENDENCIES ON)
  set(IMGUI_DAWN_DIR CACHE PATH "Path to Dawn repository")
  if(NOT IMGUI_DAWN_DIR)
    message(FATAL_ERROR "Please specify the Dawn repository by setting IMGUI_DAWN_DIR")
  endif()

  option(DAWN_FETCH_DEPENDENCIES "Use fetch_dawn_dependencies.py as an alternative to using depot_tools" ON)

  # Dawn builds many things by default - disable things we don't need
  option(DAWN_BUILD_SAMPLES "Enables building Dawn's samples" OFF)
  option(TINT_BUILD_CMD_TOOLS "Build the Tint command line tools" OFF)
  option(TINT_BUILD_DOCS "Build documentation" OFF)
  option(TINT_BUILD_TESTS "Build tests" OFF)
  if(NOT APPLE)
    option(TINT_BUILD_MSL_WRITER "Build the MSL output writer" OFF)
  endif()
  if(WIN32)
    option(TINT_BUILD_SPV_READER "Build the SPIR-V input reader" OFF)
    option(TINT_BUILD_WGSL_READER "Build the WGSL input reader" ON)
    option(TINT_BUILD_GLSL_WRITER "Build the GLSL output writer" OFF)
    option(TINT_BUILD_GLSL_VALIDATOR "Build the GLSL output validator" OFF)
    option(TINT_BUILD_SPV_WRITER "Build the SPIR-V output writer" OFF)
    option(TINT_BUILD_WGSL_WRITER "Build the WGSL output writer" ON)
  endif()

  add_subdirectory("${IMGUI_DAWN_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/dawn" EXCLUDE_FROM_ALL)

  set(LIBRARIES webgpu_dawn webgpu_cpp webgpu_glfw glfw)
endif()

# Dear ImGui
set(IMGUI_DIR "${THIRD_PARTY_DIR}/imgui")

add_library(glfw_wgpu_imgui
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_wgpu.cpp
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  # ${IMGUI_DIR}/implot/implot.h
  # ${IMGUI_DIR}/implot/implot_internal.h
  ${IMGUI_DIR}/implot/implot.cpp
  ${IMGUI_DIR}/implot/implot_items.cpp
)

if(NOT EMSCRIPTEN)
  target_compile_definitions(glfw_wgpu_imgui PUBLIC
    "IMGUI_IMPL_WEBGPU_BACKEND_DAWN"
  )
endif()

if(EMSCRIPTEN)
  if("${IMGUI_EMSCRIPTEN_GLFW3}" STREQUAL "--use-port=contrib.glfw3")
    target_compile_options(glfw_wgpu_imgui PUBLIC
      "${IMGUI_EMSCRIPTEN_GLFW3}"
    )
  endif()
  message(STATUS "Using ${IMGUI_EMSCRIPTEN_GLFW3} GLFW implementation")
  # todo: limit module fucntions https://emscripten.org/docs/api_reference/module.html#compilation-settings
  target_link_options(glfw_wgpu_imgui PUBLIC
    "-sUSE_WEBGPU=1"
    "${IMGUI_EMSCRIPTEN_GLFW3}"
    "-sWASM=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sNO_EXIT_RUNTIME=0"
    "-sASSERTIONS=1"
    "-sDISABLE_EXCEPTION_CATCHING=1"
    "-lwebsocket.js"
    "--embed-file"
    "${THIRD_PARTY_DIR}/fonts/fonts/ttf/@/fonts/"

  )
  # "-sNO_FILESYSTEM=1"

  set_target_properties(glfw_wgpu_imgui PROPERTIES OUTPUT_NAME "imgui")

endif()

target_include_directories(glfw_wgpu_imgui PUBLIC
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
  ${IMGUI_DIR}/implot
)

target_link_libraries(glfw_wgpu_imgui PUBLIC ${LIBRARIES})

################################################################################
# Modular Tracing GUI (MOTR-GUI)
# Client main executable
################################################################################
add_executable(motr_gui
  ../common/MString.cpp
  ../common/StringLibrary.cpp
  ../common/TagLibrary.cpp
  DataExporter.cpp
  DragAndDrop.cpp
  EventTreeProcessor.cpp
  FlameGraphRenderer.cpp
  GlobalState.cpp
  ImGuiContext.cpp
  LayoutLibrary.cpp
  LayoutLoader.cpp
  LayoutNode.cpp
  LayoutNodeJson.cpp
  ModularTraceSpan.cpp
  PlotLayoutNode.cpp
  RefLayoutNode.cpp
  RenderCore.cpp
  RenderPasses/RenderPasses.cpp
  RenderPasses/DrawEventTree.cpp
  RenderPasses/RenderEventTreeView.cpp
  RenderView.cpp
  RenderWindow.cpp
  TrianglePipeline.cpp
  TriangleWindow.cpp
  WebSocket.cpp
  WebUtilities.cpp
  main.cpp
  attr/AttributeResolver.cpp
  attr/DynamicValue.cpp
  attr/Convert.cpp
  attr/ExpressionEval.cpp
  attr/Parse.cpp
  attr/YogaApply.cpp
)

target_compile_options(motr_gui PUBLIC
  -Werror=switch
  -Werror=return-type
)

target_link_libraries(motr_gui PUBLIC
  glfw_wgpu_imgui
  glm::glm
  yogacore
)

target_include_directories(motr_gui PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${THIRD_PARTY_DIR}/json/single_include
  ${THIRD_PARTY_DIR}/fmt/include
  ${THIRD_PARTY_DIR}/rapidyaml/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
  ${tweeny_SOURCE_DIR}/include
)

target_compile_definitions(motr_gui PUBLIC
  MOTR_ENABLED=1
)

if(EMSCRIPTEN)
  # Export WASM/C functions for the javascript side
  target_link_options(motr_gui PUBLIC
    "-sEXPORTED_FUNCTIONS=['_main', '_handleJsonData', '_handleYamlData', '_handleLayoutJson']"
    "-sEXPORTED_RUNTIME_METHODS=['ccall']"
    "-sFETCH=1"
  )

  # Symbolic link allows for the web files to be updated without rebuilding the client
  set(WEB_FILES
    index.html
    index.css
    index.js
    layout.json
    favicon.ico
    layouts
  )

  foreach(file IN LISTS WEB_FILES)
    add_custom_command(TARGET motr_gui POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${CMAKE_CURRENT_LIST_DIR}/web/${file}"
      "$<TARGET_FILE_DIR:motr_gui>/${file}"
    )
  endforeach()

  add_custom_command(TARGET motr_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_LIST_DIR}/web/js"
    "$<TARGET_FILE_DIR:motr_gui>/js"
  )

  # Symbolic link of source directries to allow chrome dev tools
  # to load original source from DWARF debug info
  add_custom_command(TARGET motr_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${EMSCRIPTEN_ROOT_PATH}/"
    "$<TARGET_FILE_DIR:motr_gui>/emscripten"
  )

  add_custom_command(TARGET motr_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_LIST_DIR}/../../src"
    "$<TARGET_FILE_DIR:motr_gui>/src"
  )

  add_custom_command(TARGET motr_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
    "$<TARGET_FILE_DIR:motr_gui>/emsdk"
  )

  set(base_path
    "$<TARGET_FILE_DIR:motr_gui>/$<TARGET_FILE_BASE_NAME:motr_gui>")
  add_custom_command(TARGET motr_gui POST_BUILD
    COMMAND emdwp
    -e "${base_path}.wasm"
    -o "${base_path}.wasm.dwp"
  )
endif()

# Convenience debug function to print all cmake variables
function(print_variables)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

# print_variables()
