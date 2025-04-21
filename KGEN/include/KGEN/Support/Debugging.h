//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGGING_H
#define SUPPORT_DEBUGGING_H

namespace M {

/// Assuming that VS Code and the Mojo extension are active, this starts a debug
/// session that attaches to the current program. Once the attaching succeeds,
/// the debugger might auto-resume, because of which it's recommended to place
/// breakpoints before starting the debug session.
///
/// In the case in which the remote debug session fails to launch, the current
/// process will suspend itself, giving the chance for the developer to manually
/// attach to it. An appropriate message will be printed.
///
/// Example:
///
/// ```
///   void my_function() {
///     do something...
///     attachToNewRemoteDebugSession();
///     do something else...
///   }
///
/// This uses `mojo debug --vscode` under the hood.
///
/// ```
///
/// The optional parameter quiet suppresses the output of the child process.
void attachToNewRemoteDebugSession(bool quiet = false);
} // namespace M

#endif // SUPPORT_DEBUGGING_H
