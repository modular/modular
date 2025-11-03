//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_NANOBIND_PYTHONBACKTRACE_H
#define SUPPORT_NANOBIND_PYTHONBACKTRACE_H

#include "nanobind/nanobind.h"

namespace nb = nanobind;

namespace M {

using namespace M;

/// This is a header-only utility to print the Python backtrace from within C++
/// nanobind code. To use this, add `//SDK:Support` to the deps of your
/// particular `modular_nanobind_library` target. This method prints to stdout.
/// It is safe to call this method even if you do not hold the GIL. It will grab
/// it automatically.
/// While you are in lldb, you can also use `_Py_DumpTraceback(1, tstate)`
/// instead.
inline void printPythonBacktrace() {
  nb::gil_scoped_acquire gil;
  auto printStack = nb::module_::import_("traceback").attr("print_stack");
  printStack();
}

} // namespace M

#endif // SUPPORT_NANOBIND_PYTHONBACKTRACE_H
