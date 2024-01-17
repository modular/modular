# Change Log

## Next version

### Added

- The Mojo Language Server now implements the References request. IDEs use
  this to provide support for **Go to References** and **Find All References**.
  A current limitation is that references outside of the current document are
  not supported, which will be addressed in the future.

### Changed

- The initialization messages of every debug session are now suppressed in the
  Debug Console unless they fail.

### Fixed

- [#1299](https://github.com/modularml/mojo/issues/1299) - The `Run Mojo File`
  action now supports file paths with spaces.

- The Python-based LLDB visualizers in the SDK are now only enabled if
  Python-scripting support is enabled and fully functional in LLDB.

## 0.6.0

### Added

- Enabled a smarter Debug Console. It will try to determine if the input is an
  LLDB command or an expression and resolve it accordingly. The user can prepend
  the input with a colon (`:`) to force handling it as an LLDB command, which is
  similar to the behavior of the Mojo REPL.

- Mojo code blocks within documentation strings are now syntax highlighted.

- Added support for JIT debugging Mojo source files using F5 if no launch
  configurations are present.

- Added a `Debug Mojo File in Terminal` action.

- Added a `Mojo` submenu for Mojo files in the File Explorer that exposes the
  `Run or Debug Mojo File` actions.

- Added dynamic debug configurations for Mojo files, which can be accessed by
  selecting `Mojo...` in the Run and Debug Tab selector.

- Show local variables on the documents when debugging.

### Changed

- Enhanced the `Run or Debug Mojo File` actions, fixing bugs, improving icons
  and wording, as well as polishing the overall experience.
