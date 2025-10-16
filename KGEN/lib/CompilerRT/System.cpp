//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Memory.h"
#include "Support/SymbolExport.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include <cstdarg>
#include <thread>

//===----------------------------------------------------------------------===//
// CPU Information
//===----------------------------------------------------------------------===//

/// Returns the number of physical cores in the CPU, across all sockets
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_NumPhysicalCores() {
  return M::getNumPhysicalCores();
}

/// Returns the number of system threads, including hyperthreads across all
/// sockets
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_NumLogicalCores() {
  return M::getNumLogicalCores();
}

/// Returns the number of physical performance cores if the info is available,
/// otherwise returns the total number of physical cores
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_NumPerformanceCores() {
  return M::getNumPerformanceCores();
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT int
KGEN_CompilerRT_fprintf(FILE *stream, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int result = vfprintf(stream, format, args);
  va_end(args);
  return result;
}

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

namespace {
/// This class represents the set of argv values passed to the current mojo
/// program.
struct ArgVList {
  /// The raw list of arguments, used when communicating with Mojo.
  struct RawList {
    llvm::StringRef *args;
    size_t size;
  };

  /// Return the global argv instance.
  static ArgVList &get() {
    static ArgVList argVList;
    return argVList;
  }

  /// Return the raw list of arguments.
  RawList getRawList() { return {args.data(), args.size()}; }

  /// Allow argv[0] to be empty by default, matching python behavior when no
  /// script name is passed.
  std::vector<llvm::StringRef> args{""};
  std::vector<std::string> argStrings;
};
} // namespace

COMPILERRT_EXPORT
COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_GetArgV(ArgVList::RawList *result) {
  *result = ArgVList::get().getRawList();
}

COMPILERRT_EXPORT
COMPILERRT_VISIBILITY_EXPORT void KGEN_CompilerRT_SetArgV(int argc,
                                                          char **argv) {
  ArgVList &argVList = ArgVList::get();
  argVList.args.resize(argc);
  argVList.argStrings.resize(argc);
  for (int i = 0; i < argc; ++i)
    argVList.args[i] = argVList.argStrings[i] = argv[i];
}

//===----------------------------------------------------------------------===//
// Stack trace
//===----------------------------------------------------------------------===//

namespace {
std::string getStackTrace(unsigned depth = 0, unsigned dropFirst = 0) {
  std::string stacktrace;
  llvm::raw_string_ostream os(stacktrace);
  llvm::sys::PrintStackTrace(os, depth);
  return stacktrace;
}

// Return true if we should collect stack trace. Assume that no need to collect
// stack trace if `envVar` is set to '0' or 'false' (for performance, check
// first char only). If `envVar` is not set, return `defaultValue`.
bool isStackTraceEnabled(const char *envVar, bool defaultValue = false) {
  auto enableStackTrace = llvm::sys::Process::GetEnv(envVar).value_or("");
  if (enableStackTrace.empty())
    return defaultValue;
  return enableStackTrace[0] != '0' && enableStackTrace[0] != 'f' &&
         enableStackTrace[0] != 'F';
}
} // namespace

COMPILERRT_EXPORT
COMPILERRT_VISIBILITY_EXPORT void KGEN_CompilerRT_PrintStackTraceOnFault() {
  if (!isStackTraceEnabled("MOJO_ENABLE_STACK_TRACE_ON_CRASH", true)) {
    return;
  }
  llvm::sys::PrintStackTraceOnErrorSignal("", false);
}

COMPILERRT_EXPORT
COMPILERRT_VISIBILITY_EXPORT int KGEN_CompilerRT_GetStackTrace(char **strings,
                                                               unsigned depth) {
  // Cache the result of the environment variable check:
  //  -1: environment variable was not processed yet
  //   0: environment variable was processed previously and was set to '0' or
  //      'false'. Stack trace is disabled.
  //   1: environment variable was processed previously. Stack trace is enabled.
  static std::atomic<int> enabled(-1);

  int isEnabled = enabled.load();
  if (isEnabled == -1) {
    isEnabled = isStackTraceEnabled("MOJO_ENABLE_STACK_TRACE_ON_ERROR") ? 1 : 0;
    enabled.store(isEnabled);
  }

  if (isEnabled == 0)
    return 0;

  std::string stacktrace = getStackTrace(depth);
  const size_t len = stacktrace.length() + 1; // include \0 terminator
  *strings = (char *)KGEN_CompilerRT_AlignedAlloc(0, len);
  memcpy(*strings, stacktrace.c_str(), len);
  return len;
}
