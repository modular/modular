//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Utilities to help C++ debuggers
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGGER_H
#define SUPPORT_DEBUGGER_H

namespace M {

struct Debugger {
  /// Print the executable name and process ID to standard output, then pause
  /// for the specified number of seconds (default to 30 seconds) and wait. This
  /// should allow a debugger to be attached.
  static void attach(int timeoutSeconds = 30);
};

} // namespace M

#endif // SUPPORT_DEBUGGER_H
