//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/SymbolExport.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/StringRef.h"
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
