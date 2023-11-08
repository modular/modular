# 0.6.0

## Added

- Enabled a smarter Debug Console. It will try to determine if the input is an
  LLDB command or an expression and resolve it accordingly. The user can prepend
  the input with a colon (`:`) to force handling it as an LLDB command, which is
  similar to the behavior of the Mojo REPL.

- Fixed the "Debug Mojo File" action in the editor's top menu.
