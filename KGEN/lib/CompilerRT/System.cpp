//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include <cstdarg>
#include <stdio.h>
#include <thread>

//===----------------------------------------------------------------------===//
// CPU Information
//===----------------------------------------------------------------------===//

/// Returns the number of cores on the system.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_CoreCount() {
  return std::thread::hardware_concurrency();
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
// CompilerRT Registration
//===----------------------------------------------------------------------===//

void M::KGEN::registerSystem(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_CoreCount", (void *)&KGEN_CompilerRT_CoreCount});
  funcs.push_back(
      {"KGEN_CompilerRT_GetArgV", (void *)&KGEN_CompilerRT_GetArgV});
  funcs.push_back(
      {"KGEN_CompilerRT_SetArgV", (void *)&KGEN_CompilerRT_SetArgV});
}
