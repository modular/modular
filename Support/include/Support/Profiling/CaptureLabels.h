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
//
// Op labels recorded while a device graph is being captured.
//
// A replayed device graph runs no host code, so the op-level spans that name
// its kernels exist only at capture time. While a capture is in progress the
// op spans push their names here, the device launch site reads the innermost
// one into each recorded launch, and the capture keeps the ordered list of
// pushed ops. A profiler can then resolve replayed kernels back to ops.
//
// Independent of whether a profiler trace is live: captures usually run
// during warmup, before any trace starts. Outside a capture, pushing a label
// costs one relaxed atomic load.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_CAPTURELABELS_H
#define SUPPORT_PROFILING_CAPTURELABELS_H

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace M::Profiling {

namespace Detail {
std::atomic<int> &getActiveCaptures() noexcept;
} // namespace Detail

// True while at least one device-graph capture is in progress.
inline bool isCaptureLabelingActive() noexcept {
  return Detail::getActiveCaptures().load(std::memory_order_relaxed) > 0;
}

// Bracket a device-graph capture. Nestable and callable from any thread;
// the recorded op list is cleared when the outermost capture ends.
void beginCaptureLabeling();
void endCaptureLabeling();

// Push an op label on the calling thread. Returns true iff it was pushed
// (a capture is in progress); the caller pops exactly the pushes that
// returned true, on the same thread.
bool pushCaptureLabel(llvm::StringRef label);
void popCaptureLabel();

// The innermost label on the calling thread and its op sequence number, or
// an empty label and 0 when none is pushed.
struct CaptureLabel {
  std::string label;
  uint64_t opSeq = 0;
  uint64_t thread = 0;
};
CaptureLabel currentCaptureLabel();

// Carry the calling thread's innermost op to work it hands to other threads:
// read a token here, and adopt it on the worker for the work's duration. A
// launch recorded under an adopted op is attributed to that op exactly as if
// the original thread had made it. The token is 0 when no op is pushed.
uint64_t currentCaptureOp();
// Returns true iff the op was adopted (a capture is in progress and the token
// names one of its ops); the caller pops exactly the adoptions that returned
// true with popCaptureLabel(), on the same thread.
bool adoptCaptureOp(uint64_t opSeq);

// The sequence number the next pushed op will receive. Bracketing a capture
// with two reads bounds the ops pushed during it.
uint64_t nextCaptureOpSeq();

// One pushed op: `parentSeq` is the op enclosing it on the same thread (0 at
// the top level), `thread` a small process-unique id for the pushing thread,
// or for the thread that pushed the enclosing op when that op was adopted.
struct CaptureOp {
  uint64_t opSeq = 0;
  uint64_t parentSeq = 0;
  uint64_t thread = 0;
  std::string label;
};

// The ops pushed on `thread` with sequence numbers in [minSeq, maxSeq], in
// push order.
std::vector<CaptureOp> capturedOps(uint64_t thread, uint64_t minSeq,
                                   uint64_t maxSeq);

} // namespace M::Profiling

#endif // SUPPORT_PROFILING_CAPTURELABELS_H
