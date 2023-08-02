//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_CRASHPAD_H
#define KGEN_TOOLS_MOJO_CRASHPAD_H

namespace M {

/// Initialize crash reporting for Mojo driver.
void initCrashpad(const char *argv0);

} // namespace M

#endif // KGEN_TOOLS_MOJO_CRASHPAD_H
