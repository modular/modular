# 0.6.0

## Added

- Enabled a smarter Debug Console. It will try to determine if the input is an
  LLDB command or an expression and resolve it accordingly. The user can prepend
  the input with a colon (`:`) to force handling it as an LLDB command, which is
  similar to the behavior of the Mojo REPL.

- Mojo code blocks within documentation strings are now syntax highlighted.

- Added support for JIT debugging Mojo source files using F5 if no launch
  configurations are present.

## Fixed

- Fixed the `Debug Mojo File` action in the editor's top menu and drop the
  experimental tag.

- Improved the experience of the `Run Mojo File` actions.
