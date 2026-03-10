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
  Context *ptr = getCurrentMaxContextPointerOrNull();
  assert(ptr != nullptr &&
         "getCurrentMaxContext() returned nullptr; M::Context should be set at "
         "creation (Init::createContext) and cleared only in ~Context()");
  return ContextRef::copy(ptr);
}

Context *getCurrentMaxContextOrNull() {
  return getCurrentMaxContextPointerOrNull();
}

void setCurrentMaxContext(Context *ptr) { setCurrentMaxContextPointer(ptr); }

Context::~Context() { clearGlobalContextPointerIfEquals(this); }

} // namespace M
