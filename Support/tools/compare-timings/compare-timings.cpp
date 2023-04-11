//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Compare two outputs of mt's --save-timings option.
//
//===----------------------------------------------------------------------===//

#include "Support/Benchmark/Stats.h"
#include "Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::Stats;

namespace {

struct CompareTimingsCLOptions {
  llvm::cl::opt<std::string> a{
      "a", llvm::cl::desc("Path to timings output for the 'A' side"),
      llvm::cl::init(""), llvm::cl::Required};

  llvm::cl::opt<std::string> b{
      "b", llvm::cl::desc("Path to timings output for the 'B' side"),
      llvm::cl::init(""), llvm::cl::Required};
};

} // namespace

#define BIND(var, expr)                                                        \
  auto var##OrErr = (expr);                                                    \
  if (const auto *err = var##OrErr.getError()) {                               \
    llvm::errs() << err << "\n";                                               \
    return 1;                                                                  \
  }                                                                            \
  auto(var) = *var##OrErr;

int main(int argc, char **argv) {
  CompareTimingsCLOptions options;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Compare timings tool");

  BIND(aSamples, Samples::load("ns", options.a.getValue()));
  BIND(aNormal, Normal::fromSamples(aSamples));
  size_t aPruned = aSamples.prune(0.5 * aNormal.mean, 1.5 * aNormal.mean);
  BIND(aHistogram, Histogram::fromSamples(aSamples, 1e6));

  BIND(bSamples, Samples::load("ns", options.b.getValue()));
  BIND(bNormal, Normal::fromSamples(bSamples));
  size_t bPruned = bSamples.prune(0.5 * bNormal.mean, 1.5 * bNormal.mean);
  BIND(bHistogram, Histogram::fromSamples(bSamples, 1e6));

  llvm::outs() << "A:\n";
  llvm::outs() << "  pruned:   " << aPruned << "\n";
  aSamples.printSummary();
  aNormal.printSummary();
  llvm::outs() << "  histogram:\n";
  aHistogram.printSummary();
  llvm::outs() << "\n";

  llvm::outs() << "B:\n";
  llvm::outs() << "  pruned:   " << bPruned << "\n";
  bSamples.printSummary();
  bNormal.printSummary();
  llvm::outs() << "  histogram:\n";
  bHistogram.printSummary();
  llvm::outs() << "\n";

  llvm::outs() << "Speedup of B w.r.t. A:\n";
  BIND(ratioSamples,
       Samples::ratio(
           aSamples, bSamples,
           /*numSamples=*/
           std::min(aSamples.numSamples() * bSamples.numSamples(), 25000lu)));
  ratioSamples.printSummary();
  BIND(ratioHisogram, Histogram::fromSamples(ratioSamples, 0.01));
  llvm::outs() << "  histogram:\n";
  ratioHisogram.printSummary();
  llvm::outs() << "\n";

  llvm::outs() << "Welch t-test:\n";
  double welchPercentile =
      welchTTest(aSamples, aNormal, bSamples, bNormal, /*numSamples=*/1000);
  llvm::outs() << "  %ile:       " << llvm::format("%.2f", welchPercentile)
               << "%\n";
  if (welchPercentile <= 5.0)
    llvm::outs() << "  <<<B APPEARS FASTER THAN A>>>\n";
  else if (welchPercentile >= 95.0)
    llvm::outs() << "  <<<A APPEARS FASTER THAN B>>>\n";
  llvm::outs() << "\n";

  return 0;
}
