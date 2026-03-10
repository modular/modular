//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Context.h"

#include <cassert>

namespace M {

static Context *&getCurrentMaxContextInTLS() {
  static thread_local Context *currentContext = nullptr;
  return currentContext;
}

Context *getCurrentMaxContext() { return getCurrentMaxContextInTLS(); }

void setCurrentMaxContext(Context *context) {
  assert((context == nullptr || getCurrentMaxContextInTLS() == nullptr) &&
         "Max context already set; clear with setCurrentMaxContext(nullptr) "
         "before setting again");
  getCurrentMaxContextInTLS() = context;
}

} // namespace M
