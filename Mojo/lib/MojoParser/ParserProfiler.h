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
// Records the self time of declaration resolution for `--parser-stats`.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARSERPROFILER_H
#define KGEN_MOJOPARSER_PARSERPROFILER_H

#include "Mojo/MojoParser/ParserStats.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <chrono>

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;

/// Keeps a stack of (declaration, activity) frames. Each time a frame starts or
/// ends, the time since the last change goes to the frame on top. Thus a
/// frame gets only its self time, and the times add up to the wall time.
class ParserProfiler {
public:
  explicit ParserProfiler(SharedState &shared);

  void push(ASTDecl &decl, ParserActivity activity);
  void pop();

  /// Stops the clock and returns the report.
  ParserStatsPart finish(llvm::StringRef label);

  /// Records one frame for its lifetime. Does nothing for a null profiler, the
  /// state when `--parser-stats` is off.
  class Scope {
  public:
    Scope(ParserProfiler *profiler, ASTDecl &decl, ParserActivity activity)
        : profiler(profiler) {
      if (profiler)
        profiler->push(decl, activity);
    }
    ~Scope() {
      if (profiler)
        profiler->pop();
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    ParserProfiler *profiler;
  };

private:
  using Clock = std::chrono::steady_clock;

  /// Gives the time since the last change to the frame on top.
  void charge(Clock::time_point now);

  /// Returns the row of `decl`, and makes it on the first use.
  unsigned getOrCreateRow(ASTDecl &decl);

  struct Frame {
    unsigned row;
    ParserActivity activity;
  };

  SharedState &shared;
  Clock::time_point start;
  Clock::time_point last;
  llvm::SmallVector<Frame> stack;
  llvm::DenseMap<ASTDecl *, unsigned> rowIndex;
  std::vector<ParserStatsRow> rows;
  int64_t unattributedNanos = 0;
  int64_t overheadNanos = 0;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSERPROFILER_H
