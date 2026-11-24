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
// The records disableCollecting() (Ranges.h) hands a stopped trace back in,
// for a caller that reads the trace in process instead of from its file.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TRACEACTIVITY_H
#define SUPPORT_PROFILING_TRACEACTIVITY_H

#include <cstdint>

#include "llvm/ADT/StringRef.h"

namespace M::Profiling {

// One activity of a stopped trace: a host range, a runtime call, or device
// work. The fields are the trace file's, unconverted. The strings are valid
// only during the visitor call that carries them.
struct TraceActivity {
  llvm::StringRef name;
  // libkineto's activity type name, as the trace file's `cat` field spells
  // it ("user_annotation", "kernel", "gpu_memcpy", "cuda_runtime", ...).
  llvm::StringRef category;
  // system_clock nanoseconds since epoch.
  int64_t startNs;
  int64_t durationNs;
  // The device and stream for device work; the process and thread track for
  // host activities.
  int64_t device;
  int64_t resource;
  int32_t threadId;
  int64_t correlationId;
  // The range the activity is correlated to, as the trace file's
  // `External id`; 0 when none.
  int64_t externalId;
};

// A top-level metadata field of a stopped trace, with its value as raw JSON.
struct TraceMetadataField {
  llvm::StringRef key;
  llvm::StringRef jsonValue;
};

} // namespace M::Profiling

#endif // SUPPORT_PROFILING_TRACEACTIVITY_H
