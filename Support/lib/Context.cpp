//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Context.h"
#include "Support/ContextGlobal.h"

#include <cassert>

namespace M {

Context::~Context() {
  std::lock_guard<std::mutex> lock(getGlobalContextMutex());
  clearGlobalContextPointerIfEquals(this);
}

} // namespace M
