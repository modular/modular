# MOdular TRacing (MOTR)

Welcome to the Modular Tracing System, a.k.a. **MOTR**!

## Planning

- Linear: [Modular Tracing System Project](https://linear.app/modularml/project/tracing-system-2025-4a76b4046632)
- Notion: [MOTR Roadmap](https://www.notion.so/modularai/MOTR-Roadmap-1a11044d37bb80abb624ea851178940d?pvs=4)

## Components

- MOTR API - C++ API header-only '#include "motr/motr.h"
- `motr server` - A server for storing and serving traces
- `motr flags <set|get|list>` - A tool for managing MAX configuration
- MOTR GUI - A web-based WASM/WebGPU viewer for the Modular Tracing System

### MOTR Server

- C++ command line tool
- Serves MOTR GUI on `localhost:8888`

### MOTR GUI

Tech Stack:

- C++ -> WASM (via Emscripten `v3.1.74`) -> HTML/JS/CSS
- Dear ImGui `v1.91.7-docking` based off [example_glfw_wgpu/ code](https://github.com/ocornut/imgui/tree/c0ae3258f9da5b959b214213ffe37ab36ea7f76f/examples/example_glfw_wgpu).
- WebGPU for all rendering
- Websockets for messaging with `motr server`
- GLFW, Nlohmann JSON, RapidYAML, fmt
- Can be cross-compiled to native for desktop (Linux, Windows, MacOS) (not yet tested)

Planned Features:

- MOTR Interactive Tracing Viewer (WebGPU flame graph)
  - Zoomamble Flame Graph
  - Real time event streaming
- MAX Configuration Editor
  - View/Edit/Save/Load MAX configuration parameters
  - Drag and drop load MAX YMAL configuration files
  - Edit MAX configuration
  - Save MAX configuration
  - Load MAX configuration
  - Validate MAX configuration

## Building

### `motr` CLI

```bash
# debug build (default)
./build-cli.sh

# release build
CMAKE_BUILD_TYPE=Release ./build-cli.sh

# clean debug build (deletes build/ dir)
CLEAN=1 ./build-cli.sh # clean build
```

### MOTR GUI

```bash
./build-gui-web.sh
CMAKE_BUILD_TYPE=Release ./build-gui-web.sh
CLEAN=1 ./build-gui-web.sh
```
