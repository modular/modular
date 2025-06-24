# Modular Tracing (MOTR) Project Overview

## 1. `motr` Command Line Tool

- **Functionality**: Provides server-side command-line utilities for managing
  and interacting with the MOTR tracing system. This includes:
  - `motr server`: Core tracing process controller provides:
    - Trace data collection via Shared Memory (SHM)
    - MOTR Gui client serving via HTTP
    - Trace data streaming to MOTR Gui client via Websockets
  - `motr <flags,strings,tags,test>`: Command line tools to manipulate the SHM
    state of running MOTR processes
- **Code Locations**: `src/cli` and `src/common` dirs.

## 2. MOTR Gui Client

- **Functionality**: A WebAssembly (WASM) and WebGPU based graphical user
  interface that provides real-time visualization and interaction of distributed
  tracing data. It includes components for rendering, event processing, and
  user interaction.
  - Distributed tracing communication is done with multiple `motr server`
    processes via websocket connections to `http://localhost:[6680-6687]/ws`
- **Code Locations**: `src/gui` and `src/common` dirs.

## 3. Common MOTR API

- **Functionality**: A C++ based API that facilitates integration with the
  broader Modular codebase.
  - Provides C++ header-only implementations of trace instrumentation and
    shared-memory (SHM) messaging
- **Code Locations**: `src/common` dir

## 4. Dependencies

- [CivetWeb](https://github.com/civetweb/civetweb): HTTP server and WebSocket support.
- [nlohmann JSON](https://github.com/nlohmann/json): JSON parsing and serialization.
- [Dear ImGui](https://github.com/ocornut/imgui): GUI library for tools and applications.
- [{fmt}](https://github.com/fmtlib/fmt): Modern C++ formatting library.
- [Emscripten](https://emscripten.org/): Compiles C++ to WebAssembly.
- [WebGPU](https://www.w3.org/TR/webgpu/): High-performance graphics rendering.
- [CMake](https://cmake.org): Build process management.
- [Yoga](https://github.com/facebook/yoga): Flexbox-based layout engine.
