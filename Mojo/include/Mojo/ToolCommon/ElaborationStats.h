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
// The `--elaboration-stats` report: what each `comptime for` loop produced,
// and what the interpreter calls for comptime values cost. The elaborators
// record it. A driver that collects it prints every pipeline's part through the
// timing sink, as text or JSON; without a collector, an elaboration prints its
// own text report to stderr, which is what `kgen-opt` wants.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_ELABORATIONSTATS_H
#define KGEN_TOOLCOMMON_ELABORATIONSTATS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <vector>

namespace M::KGEN {
class CompilationOptions;

/// What elaborating one `hlcf.comptime.for` produced.
struct ComptimeForRecord {
  mlir::Location loc;
  /// The generator whose instantiation held the loop.
  mlir::StringAttr generator;
  int64_t iterations;
  /// The operations in the body before unrolling.
  int64_t bodyOps;
  /// The operations in the unrolled iterations once they elaborated, before
  /// inlining.
  int64_t resultOps;
};

/// One call of the interpreter for a comptime value.
struct InterpreterCallRecord {
  /// The operation whose comptime value needed the call.
  mlir::Location loc;
  /// The concrete function the call runs.
  mlir::StringAttr callee;
  /// The CPU time the calling thread spent interpreting, unset when the
  /// elaborator's cache had the value.
  std::optional<int64_t> cpuNanos;
};

/// The records of one elaboration, under the name of its pipeline, such as
/// `host x86_64-linux-gnu`, the name the timing reports use.
struct ElaborationStatsPart {
  std::string label;
  std::vector<ComptimeForRecord> comptimeForRecords;
  /// Unset for the parametric elaborator, which does not record its calls.
  std::optional<std::vector<InterpreterCallRecord>> interpreterCalls;
};

/// Returns the CPU time the calling thread has used, in nanoseconds.
int64_t getThreadCPUNanos();

/// Starts to collect the parts for a report the driver prints. The driver
/// calls this for the `--elaboration-stats` option, before the compilation
/// starts.
void enableElaborationStats();

/// Takes the collected parts and stops the collection. A pipeline that the
/// compilation cache answered did not elaborate and gives no part.
llvm::SmallVector<ElaborationStatsPart> takeElaborationStats();

/// Reports one elaboration: a part for the collection when it is on, else a
/// text report on stderr.
void reportElaborationStats(bool isOffload, const CompilationOptions &options,
                            std::vector<ComptimeForRecord> comptimeForRecords,
                            std::optional<std::vector<InterpreterCallRecord>>
                                interpreterCalls = std::nullopt);

/// Prints `part` as text. Each section is a summary line, then its rows: the
/// loops grouped by location and generator, sorted by the operations they
/// produced, and the interpreter calls grouped by callee, then by location and
/// callee, both sorted by CPU time.
void printElaborationStats(llvm::raw_ostream &os,
                           const ElaborationStatsPart &part);

/// Writes `parts` as one JSON array, one object per pipeline.
void writeElaborationStatsJSON(llvm::raw_ostream &os,
                               llvm::ArrayRef<ElaborationStatsPart> parts);

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_ELABORATIONSTATS_H
