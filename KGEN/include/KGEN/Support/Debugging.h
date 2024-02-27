//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGGING_H
#define SUPPORT_DEBUGGING_H

namespace M {
/// Assuming that VS Code and the Mojo extension are active, this starts a debug
/// session attaching to the current program on the IDE. Once the debugger
/// attaches to current program, it might auto resume, because of which it's
/// recommended to place breakpoints before starting the debug session.
///
/// This uses `mojo debug --rpc` under the hood.
void attachToRemoteDebugger();
} // namespace M

#endif // SUPPORT_DEBUGGING_H
