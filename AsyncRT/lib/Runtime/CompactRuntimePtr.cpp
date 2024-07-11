//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/CompactRuntimePtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_os_ostream.h"

#include <mutex>

using namespace M;
using namespace M::LLCL;

Detail::RuntimeTable::RuntimeTable() {
  freeIndices.resize(kInvalidIndex);
  for (uint8_t i = 0; i < kInvalidIndex; i++) {
    allRuntimes[i] = nullptr;
    freeIndices[i] = kInvalidIndex - i - 1;
  }
}

Runtime *Detail::RuntimeTable::getRuntime(uint8_t index) const {
  assert(index != kInvalidIndex && "invalid Runtime index");
  assert(allRuntimes[index] != nullptr &&
         "no Runtime has been registered for index");
  // NOTE: We are assuming the mutex lock will force all writes to allRuntimes
  // to be flushed.
  return allRuntimes[index];
}

uint8_t Detail::RuntimeTable::reserveIndex() {
  std::lock_guard<std::mutex> lock(mu);
  assert(!freeIndices.empty() && "too many Runtimes are currently active");
  auto index = freeIndices.pop_back_val();
  assert(allRuntimes[index] == nullptr &&
         "index is still occupied by a Runtime");
  return index;
}

void Detail::RuntimeTable::setRuntime(uint8_t index, Runtime *runtime) {
  // NOTE: Take the lock to ensure writes to allRuntimes are flushed.
  std::lock_guard<std::mutex> lock(mu);
  allRuntimes[index] = runtime;
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
