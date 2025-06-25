# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from modular.utils.benchmarking import BenchmarkResult

benchmark_log = """================================================
MLPerf Results Summary
================================================
SUT name : benchmark
Scenario : SingleStream
Mode     : PerformanceOnly
90th percentile latency (ns) : 7132046
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
Early Stopping Result:
 * Processed at least 64 queries (708).
 * Would discard 51 highest latency queries.
 * Early stopping 90th percentile estimate: 7171535
 * Early stopping 99th percentile estimate: 10952369

================================================
Additional Stats
================================================
QPS w/ loadgen overhead         : 141.400
QPS w/o loadgen overhead        : 141.528

Min latency (ns)                : 6930533
Max latency (ns)                : 10952369
Mean latency (ns)               : 7065755
50.00 percentile latency (ns)   : 7023578
90.00 percentile latency (ns)   : 7132046
95.00 percentile latency (ns)   : 7252009
97.00 percentile latency (ns)   : 7340995
99.00 percentile latency (ns)   : 7577115
99.90 percentile latency (ns)   : 10952369

================================================
Test Parameters Used
================================================
samples_per_query : 1
target_qps : 1000
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 5000
max_duration (ms): 0
min_query_count : 20
max_query_count : 0
qsl_rng_seed : 0
sample_index_rng_seed : 0
schedule_rng_seed : 0
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 1

No warnings encountered during test.

No errors encountered during test.
"""


def test_from_lines() -> None:
    lines = benchmark_log.split("\n")
    res = BenchmarkResult.from_lines(lines)
    assert res.qps == 141.528
    assert res.min_latency == 6930533
    assert res.percentile_9000 == 7132046
