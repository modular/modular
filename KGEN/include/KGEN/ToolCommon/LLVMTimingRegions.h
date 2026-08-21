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
// Splits the LLVM pass timing report into one part for each pipeline that a
// compiler command runs. A command runs the host pipeline, and it runs one
// more pipeline for each `kgen.compile_offload` target.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_LLVMTIMINGREGIONS_H
#define KGEN_TOOLCOMMON_LLVMTIMINGREGIONS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M::KGEN {
class CompilationOptions;

/// One part of the LLVM pass timing report.
struct LLVMTimingReportPart {
  /// Names the pipeline that made this part, such as `host x86_64-linux-gnu`.
  std::string label;
  /// The report of LLVM for this pipeline, in the format that LLVM writes.
  std::string report;
};

/// Starts to collect the report in parts. The driver calls this for the
/// `--llvm-timing` option, before the compilation starts.
///
/// The timers of LLVM are global to the process, and LLVM gives no way to hold
/// a separate set for each pipeline. The collection instead reads the timers
/// and clears them each time the pipeline changes, which gives the times of
/// only the pipeline that ran since the previous read. This is correct because
/// the pipelines do not overlap: the option also sets the compilation to one
/// thread, and the compiler runs each offload pipeline inside elaboration,
/// which is before the host pipeline reaches LLVM.
void enableLLVMTimingRegions();

/// Reads the parts and forgets them. The parts come in the order in which the
/// pipelines ran. A pipeline whose timers hold nothing, such as one that the
/// compilation cache answered, gives no part.
llvm::SmallVector<LLVMTimingReportPart> takeLLVMTimingReport();

/// Marks the LLVM work of one pipeline. The object compiler puts one of these
/// around each piece of work that reaches LLVM. Two regions that follow each
/// other and have the same label give one part, so that a target with many
/// kernel groups does not give one part for each group. A region inside
/// another region belongs to the outer region.
///
/// The object does nothing unless `enableLLVMTimingRegions` ran first.
class LLVMTimingRegion {
public:
  explicit LLVMTimingRegion(const CompilationOptions &options);
  ~LLVMTimingRegion();

  LLVMTimingRegion(const LLVMTimingRegion &) = delete;
  LLVMTimingRegion &operator=(const LLVMTimingRegion &) = delete;

private:
  bool active;
};

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_LLVMTIMINGREGIONS_H
