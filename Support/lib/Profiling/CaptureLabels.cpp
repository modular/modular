//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Support/Profiling/CaptureLabels.h"

#include "Support/Globals/Globals.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <mutex>

namespace M::Profiling {

namespace {

Globals::CaptureLabelGlobals &globals() noexcept {
  static Globals::CaptureLabelGlobals &g = M::Globals::getCaptureLabelGlobals();
  return g;
}

} // namespace

namespace Detail {
std::atomic<int> &getActiveCaptures() noexcept {
  return globals().activeCaptures;
}
} // namespace Detail

void beginCaptureLabeling() {
  globals().activeCaptures.fetch_add(1, std::memory_order_acq_rel);
}

void endCaptureLabeling() {
  Globals::CaptureLabelGlobals &g = globals();
  if (g.activeCaptures.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  std::lock_guard<std::mutex> lock(g.opsMutex);
  g.ops.clear();
}

bool pushCaptureLabel(llvm::StringRef label) {
  if (!isCaptureLabelingActive())
    return false;
  Globals::CaptureLabelGlobals &g = globals();
  Globals::CaptureLabelThreadState &state =
      M::Globals::getCaptureLabelThreadState();
  if (state.thread == 0)
    state.thread = g.nextThread.fetch_add(1, std::memory_order_relaxed);
  uint64_t opSeq = g.nextOpSeq.fetch_add(1, std::memory_order_relaxed);
  uint64_t parentSeq = state.stack.empty() ? 0 : state.stack.back().opSeq;
  // An op nested under an adopted one belongs to the adopted op's thread, so
  // readers that group ops by thread keep the whole subtree together.
  uint64_t thread =
      state.stack.empty() ? state.thread : state.stack.back().thread;
  state.stack.push_back({label.str(), opSeq, thread});
  std::lock_guard<std::mutex> lock(g.opsMutex);
  g.ops.push_back({opSeq, parentSeq, thread, label.str()});
  return true;
}

void popCaptureLabel() {
  Globals::CaptureLabelThreadState &state =
      M::Globals::getCaptureLabelThreadState();
  if (!state.stack.empty())
    state.stack.pop_back();
}

CaptureLabel currentCaptureLabel() {
  Globals::CaptureLabelThreadState &state =
      M::Globals::getCaptureLabelThreadState();
  if (state.stack.empty())
    return {};
  const Globals::CaptureLabelFrame &top = state.stack.back();
  return {top.label, top.opSeq, top.thread};
}

uint64_t currentCaptureOp() {
  if (!isCaptureLabelingActive())
    return 0;
  return currentCaptureLabel().opSeq;
}

bool adoptCaptureOp(uint64_t opSeq) {
  if (opSeq == 0 || !isCaptureLabelingActive())
    return false;
  Globals::CaptureLabelGlobals &g = globals();
  Globals::CaptureLabelFrame frame;
  {
    std::lock_guard<std::mutex> lock(g.opsMutex);
    auto it = std::find_if(
        g.ops.rbegin(), g.ops.rend(),
        [&](const Globals::CaptureLabelOp &op) { return op.opSeq == opSeq; });
    if (it == g.ops.rend())
      return false;
    frame = {it->label, it->opSeq, it->thread};
  }
  M::Globals::getCaptureLabelThreadState().stack.push_back(std::move(frame));
  return true;
}

uint64_t nextCaptureOpSeq() {
  return globals().nextOpSeq.load(std::memory_order_relaxed);
}

std::vector<CaptureOp> capturedOps(uint64_t thread, uint64_t minSeq,
                                   uint64_t maxSeq) {
  Globals::CaptureLabelGlobals &g = globals();
  std::vector<CaptureOp> result;
  std::lock_guard<std::mutex> lock(g.opsMutex);
  for (const Globals::CaptureLabelOp &op : g.ops) {
    if (op.thread == thread && op.opSeq >= minSeq && op.opSeq <= maxSeq)
      result.push_back({op.opSeq, op.parentSeq, op.thread, op.label});
  }
  return result;
}

} // namespace M::Profiling
