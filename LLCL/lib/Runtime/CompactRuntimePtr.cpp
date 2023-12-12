//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/CompactRuntimePtr.h"

#include <mutex>

using namespace M;
using namespace M::LLCL;

namespace {
static thread_local CompactRuntimePtr currentRuntimeInTLS;
}

Detail::RuntimeTable::RuntimeTable() {
  freeIndices.resize(kInvalidIndex);
  for (uint8_t i = 0; i < kInvalidIndex; i++) {
    allRuntimes[i] = nullptr;
    freeIndices[i] = kInvalidIndex - i - 1;
  }
}

Runtime *Detail::RuntimeTable::getRuntime(uint8_t index) const {
  /// CAUTION: Not using mutex, may not see side effects from other threads!
  assert(index != kInvalidIndex && "invalid Runtime index");
  assert(allRuntimes[index] != nullptr &&
         "no Runtime has been registered for index");
  return allRuntimes[index];
}

uint8_t Detail::RuntimeTable::addRuntime(Runtime *runtime) {
  std::lock_guard<std::mutex> lock(mu);
  assert(!freeIndices.empty() && "too many Runtimes are currently active");
  auto index = freeIndices.pop_back_val();
  assert(allRuntimes[index] == nullptr &&
         "index is still occupied by a Runtime");
  allRuntimes[index] = runtime;
  return index;
}

void Detail::RuntimeTable::clearRuntime(uint8_t index) {
  std::lock_guard<std::mutex> lock(mu);
  assert(allRuntimes[index] != nullptr &&
         "no Runtime has been registered for index");
  assert(freeIndices.size() < kInvalidIndex && "all indices are already free");
  allRuntimes[index] = nullptr;
  freeIndices.push_back(index);
}

size_t Detail::RuntimeTable::numActiveRuntimes() const {
  std::lock_guard<std::mutex> lock(mu);
  return kInvalidIndex - freeIndices.size();
}

CompactRuntimePtr CompactRuntimePtr::getCurrentRuntime() {
  return currentRuntimeInTLS;
}

void CompactRuntimePtr::setCurrentRuntime(CompactRuntimePtr ptr) {
  currentRuntimeInTLS = ptr;
}
