//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Process.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;

LogicalResult M::setProcessEnv(StringRef name, StringRef value,
                               bool overwrite) {
#ifdef _WIN32
  if (!overwrite && llvm::sys::Process::GetEnv(name))
    return success();
  int result = SetEnvironmentVariableA(name.str().data(), value.str().data());
  return success(result != 0);
#else
  int result = setenv(name.str().data(), value.str().data(), overwrite);
  return success(result == 0);
#endif
}
