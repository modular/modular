//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/AlignedAlloc.h"
#include "Support/MathExtras.h"
#include "Support/MicroBenchmark.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace M;
using namespace std::chrono_literals;

namespace {
/// An opaque function containing data.
struct OpaqueFunction {
  /// Opaque data passed into each function call.
  void *data;
  /// The function implementation. The first argument is always `data` and the
  /// second is a pointer to an argument pack. Returns an opaque result pack.
  void *(*fn)(void *, void *);

  /// Invoke the function.
  void *invoke(void *args) const { return (*fn)(data, args); }
};
} // namespace

/// This function takes a nullary function and benchmarks it using the
/// micrbenchmarking framework. It returns the mean runtime of the function in
/// nanoseconds.
extern "C" double KGEN_CompilerRT_Benchmark(OpaqueFunction func) {
  MicroBenchmark benchmark("kgen benchmark function",
                           [func](MicroBenchmark::State &state) {
                             for (auto _ : state)
                               func.invoke(nullptr);
                           });

  MicroBenchmark::RunOptions runOptions;
  runOptions.printWarningIfDebugMode = false;
#ifdef MODULAR_DEBUG
  runOptions.minRuntime = 20ms;
#else  // MODULAR_DEBUG
  runOptions.minRuntime = 100ms;
#endif // MODULAR_DEBUG

  (void)benchmark.run(runOptions);
  return benchmark.measurement(
      MicroBenchmark::ReportMetric::kTrimmedMeanLatency,
      MicroBenchmark::TimeUnit::kNanoseconds);
}

/// This function takes a list of opaque functions to evaluate and opaque
/// configurations for which to evaluate them.
///
/// Each of the opaque functions should refer to a "wrapper" around a concrete
/// function pointer of an interface implementation to evaluate that unpacks
/// opaque arguments and calls the function. For example, if the interface
/// under evaluation is `kgen.generator.interface @foo(i64, i32) -> (f64, f32)`,
/// the wrapper should unpack a `void *` argument pack as `struct<i64, i32>` and
/// store to the result pack a `struct<f64, f32>`, returned as a pointer.
///
/// `fn` is a pointer to the wrapper function and `data` is a pointer to the
/// actual interface implementation.
///
/// `configs` is an array of opaque evaluation configs, which could be, for
/// instance, pointers to structs or other data used to construct function
/// arguments and evaluate their results. `weights` is an array of the same size
/// which determines the weight of each configuration. `populateFn` takes an
/// evaluation config and returns a generated argument pack. If the result is
/// `nullptr`, the funciton fails and returns `-1`.
///
/// `compareFn` is called with the arguments and results of one function call
/// and the arguments and results of another function call. It should return
/// `true` if the results are consistent and `false` otherwise. When it returns
/// `false`, this function will return `-1` to indicate a bad selection.
///
/// `argDestroyFn` is called with a config and an allocated argument pack to
/// deallocate any memory. `resDestroyFn` is called with a config and an
/// allocated result pack to deallocate any memory.
///
/// If the function ever fails, `emitErrorFn` is invoked with a basic error
/// message indicating the nature of the failure.
extern "C" ssize_t KGEN_CompilerRT_SelectFastestFunction(
    ArrayRef<OpaqueFunction> functions, ArrayRef<int64_t> weights,
    ArrayRef<void *> configs, void *(*populateFn)(void *),
    bool (*compareFn)(void *, void *, void *, void *),
    void (*argDestroyFn)(void *, void *), void (*resDestroyFn)(void *, void *),
    void (*emitErrorFn)(const char *)) {

  struct EvaluatedFunc {
    ssize_t funcIdx;
    double timing;
    int64_t weight;
  };

  // Evaluate the functions for each configuration.
  SmallVector<EvaluatedFunc> bestPerConfig;
  bestPerConfig.reserve(configs.size());
  for (auto configWeight : llvm::zip(configs, weights)) {
    void *config = std::get<0>(configWeight);
    int64_t weight = std::get<1>(configWeight);

    // Run each function once and ensure their results are consistent.
    SmallVector<std::pair<void *, void *>> argsAndResults;
    argsAndResults.reserve(functions.size());
    auto cleanup = llvm::make_scope_exit([&] {
      for (auto [args, results] : argsAndResults) {
        argDestroyFn(config, args);
        resDestroyFn(config, results);
      }
    });

    for (const OpaqueFunction &func : functions) {
      // Populate arguments for this function.
      void *args = populateFn(config);
      if (!args) {
        emitErrorFn("failed to populate inputs for function call");
        return -1;
      }

      // Invoke the function.
      void *results = func.invoke(args);
      if (!results) {
        argDestroyFn(config, args);
        emitErrorFn("function invocation failed");
        return -1;
      }

      argsAndResults.emplace_back(args, results);
    }

    // Do a basic, pair-wise comparison to make sure the results are consistent.
    for (unsigned i = 0, e = argsAndResults.size() - 1; i < e; ++i) {
      const std::pair<void *, void *> &lhs = argsAndResults[i];
      const std::pair<void *, void *> &rhs = argsAndResults[i + 1];
      if (!compareFn(lhs.first, lhs.second, rhs.first, rhs.second)) {
        emitErrorFn("function did not sufficiently match a previous run");
        return -1;
      }
    }

    // Now actually benchmark each function.
    double curBestTime = std::numeric_limits<double>::max();
    ssize_t curBestIdx = 0;
    for (ssize_t idx = 0, e = functions.size(); idx < e; ++idx) {
      const OpaqueFunction &func = functions[idx];
      void *args = argsAndResults[idx].first;

      // Store the results of each batch and deallocate them and the end of each
      // batch run.
      MicroBenchmark::RunOptions runOptions;
      SmallVector<void *> results;
      runOptions.prologueFunction = [&](MicroBenchmark::State &state) {
        results.reserve(state.getBatchSize());
      };
      runOptions.epilogueFunction = [&](MicroBenchmark::State &state) {
        for (void *res : results)
          resDestroyFn(config, res);
      };

      // Invoke the function and store its results.
      auto callback = [func, args, &results](MicroBenchmark::State &state) {
        for (auto _ : state)
          results.push_back(func.invoke(args));
      };
      MicroBenchmark benchmark("kgen benchmark function", callback);

      runOptions.printWarningIfDebugMode = false;
#ifdef MODULAR_DEBUG
      runOptions.minRuntime = 20ms;
#else  // MODULAR_DEBUG
      runOptions.minRuntime = 100ms;
#endif // MODULAR_DEBUG

      (void)benchmark.run(runOptions);
      double time = benchmark.measurement(
          MicroBenchmark::ReportMetric::kTrimmedMeanLatency,
          MicroBenchmark::TimeUnit::kNanoseconds);

      if (time < curBestTime) {
        curBestTime = time;
        curBestIdx = idx;
      }
    }

    // Append the best kernel to the list.
    bestPerConfig.push_back({curBestIdx, curBestTime, weight});
  }

  // Now figure out which kernel is actually best by seeing which one has the
  // lowest timing *and* the highest config weight.
  auto best = std::min_element(
      bestPerConfig.begin(), bestPerConfig.end(),
      [](const EvaluatedFunc &lhs, const EvaluatedFunc &rhs) {
        return lhs.timing < rhs.timing && lhs.weight > rhs.weight;
      });

  return best->funcIdx;
}

// This is the data distribution we're going to use for tuning memset.
// It is represented as a string of size-weight pairs,  separated by ';'.
// Elements in the pair are separated by ':'.
static constexpr llvm::StringLiteral KGEN_MEMSET_DATA_HIST = "18:10;15:30";

static SmallVector<std::pair<ssize_t, ssize_t>>
parseMemsetDataHistogramString() {
  SmallVector<std::pair<ssize_t, ssize_t>> result;

  for (auto size_count_pair : llvm::split(KGEN_MEMSET_DATA_HIST, ";")) {
    ssize_t size, count;
    SmallVector<StringRef> substrings(llvm::split(size_count_pair, ":"));
    llvm::to_integer(substrings[0], size);
    llvm::to_integer(substrings[1], count);
    result.push_back({size, count});
  }

  return result;
}

// This is an ad-hoc harness for benchmarking memset implementations.
// We use it for tuning memset implementation for very small sizes (<32 bytes)
// and because of that we need to ensure that the harness introduces as little
// overhead as possible. Ideally we should be using the standard
// KGEN_CompilerRT_Benchmark function for this, but currently it seems that the
// overhead of having OpaqueFunction is too high for this use case.

// TODO: eventually we still should use that function to avoid code duplication.
static constexpr size_t ITER = 10'000;
static constexpr size_t SAMPLES = 20;

using memset_ty = void(uint8_t *, uint8_t, ssize_t);

uint64_t measureScore(memset_ty *fn, ssize_t size) {
  void *ptr = M::alignedAlloc(kPreferredMemoryAlignment, size);

  std::vector<uint64_t> samples;

  for (unsigned i = 0; i < SAMPLES; i++) {

    std::chrono::steady_clock::time_point tic =
        std::chrono::steady_clock::now();

    for (size_t j = 0; j < ITER; j++)
      (fn)((uint8_t *)ptr, 0, size);

    std::chrono::steady_clock::time_point toc =
        std::chrono::steady_clock::now();
    uint64_t interval =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count();
    samples.push_back(interval);
  }
  M::alignedFree(ptr);

  // Return median
  std::sort(samples.begin(), samples.end());
  uint64_t res = median(samples);
  return res;
}

// Benchmark the given memset implementation on the given distribution of input
// data.
uint64_t
benchMemsetFn(void (*fn)(uint8_t *, uint8_t, ssize_t),
              const SmallVector<std::pair<ssize_t, ssize_t>> &inputSpec) {
  uint64_t result = 0;
  for (size_t i = 0; i < inputSpec.size(); i++) {
    auto size = inputSpec[i].first;
    auto weight = inputSpec[i].second;

    uint64_t score = measureScore(fn, size);
    result += score * weight;
  }
  return result;
}

extern "C" ssize_t KGEN_CompilerRT_SelectFastestMemset(memset_ty **fns,
                                                       ssize_t num) {
  // Make it static to avoid parsing the string every time
  static SmallVector<std::pair<ssize_t, ssize_t>> v =
      parseMemsetDataHistogramString();
  SmallVector<int> scores;
  for (ssize_t fn_idx = 0; fn_idx < num; fn_idx++) {
    auto fn = fns[fn_idx];
    int score = benchMemsetFn(fn, v);
    scores.push_back(score);
  }
  ssize_t best_idx = 0;
  for (ssize_t fn_idx = 1; fn_idx < num; fn_idx++) {
    if (scores[fn_idx] < scores[best_idx])
      best_idx = fn_idx;
  }
  return best_idx;
}
