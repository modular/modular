# GlidIR

GlidIR is an interactive MLIR visualization tool built with ImGui and OpenGL.

## Features

- Multiple files in one window: a **Files** sidebar lists every loaded IR file;
  click one to switch the views to it, each keeping its own bookmarks and graph
- Side-by-side IR text viewer and tree explorer
- Bookmarks: double-click an operation in the IR to pin it for quick navigation
- Regex search with next/previous navigation
- Tree view shows ops, regions, blocks, types, attributes, operands, and results
- Inline stats on each tree node (operand/result counts, block/op counts)
- Font size adjustment with Cmd/Ctrl +/-

## Usage

Start with no file and open one later, or pass one or more files:

```text
glidir [path/to/first.mlir path/to/second.mlir ...]
```

Open more files at runtime with **File > Open** (multi-select supported), or
drag files from your file manager onto the window. Switch between loaded files
from the **Files** sidebar; right-click an entry to close it.
