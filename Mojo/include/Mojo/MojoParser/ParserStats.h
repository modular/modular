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
// The `--parser-stats` report: the self time the parser spends resolving each
// declaration, split by activity. A time is exclusive: the resolution of a
// declaration that another resolution demands counts for that declaration
// only. Each parse gives one part. A driver that collects the parts prints
// them through the timing sink, as text or JSON.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARSERSTATS_H
#define KGEN_MOJOPARSER_PARSERSTATS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace M::KGEN::LIT {

/// The work the parser does for a declaration.
enum class ParserActivity : uint8_t {
  /// Parses the signature from source.
  Signature,
  /// Parses the body from source.
  Body,
  /// Makes the signature of a declaration that has no source.
  SyntheticSignature,
  /// Makes the body of a declaration that has no source.
  SyntheticBody,
  /// Resolves a declaration that a precompiled package holds.
  Bytecode,
};

inline constexpr size_t kNumParserActivities = 5;

/// Returns the name of `activity` in the report.
llvm::StringRef getParserActivityName(ParserActivity activity);

/// The self times of one declaration.
struct ParserStatsRow {
  /// The names of the declaration and its parents, joined with `.`.
  std::string name;
  std::string kind;
  /// `file:line:col`, which tells apart declarations of the same name.
  std::string location;
  std::array<int64_t, kNumParserActivities> nanos{};
  /// The number of times the resolver started an activity for this
  /// declaration.
  int64_t frames = 0;

  int64_t getTotalNanos() const;
};

/// The report of one parse.
struct ParserStatsPart {
  /// The module or package that the parse started from.
  std::string label;
  std::vector<ParserStatsRow> rows;
  /// The time outside the resolution of any declaration.
  int64_t unattributedNanos = 0;
  /// The time the profiler spends to record the rows.
  int64_t overheadNanos = 0;
  /// The wall time of the parse. It is the sum of every row, the unattributed
  /// time, and the overhead.
  int64_t totalNanos = 0;
};

/// Starts to collect the parts for a report the driver prints. The driver
/// calls this for the `--parser-stats` option, before it parses.
void enableParserStats();

/// Returns true if a driver collects the parts.
bool isParserStatsEnabled();

/// Takes the collected parts and stops the collection.
llvm::SmallVector<ParserStatsPart> takeParserStats();

/// Adds `part` to the collection, if it is on.
void reportParserStats(ParserStatsPart part);

/// Prints `part` as text: a summary for each activity, then the declarations
/// with the most self time.
void printParserStats(llvm::raw_ostream &os, const ParserStatsPart &part);

/// Writes `parts` as one JSON array, one object for each parse. Each object
/// lists every declaration, sorted by self time.
void writeParserStatsJSON(llvm::raw_ostream &os,
                          llvm::ArrayRef<ParserStatsPart> parts);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSERSTATS_H
