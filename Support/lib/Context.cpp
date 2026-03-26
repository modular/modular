//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Context.h"
#include "Support/ContextGlobal.h"

#include <cassert>

namespace M {

ContextRef getCurrentMaxContext() {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  Context *ptr = getCurrentMaxContextPointerOrNull();
  assert(ptr != nullptr &&
         "getCurrentMaxContext() returned nullptr; M::Context should be set at "
         "creation (Init::createContext) and cleared only in ~Context()");
  return ContextRef::copy(ptr);
}

Context *getCurrentMaxContextOrNull() {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  return getCurrentMaxContextPointerOrNull();
}

void setCurrentMaxContext(Context *ptr) {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  setCurrentMaxContextPointer(ptr);
}

Context::~Context() {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  clearGlobalContextPointerIfEquals(this);
}

} // namespace M
