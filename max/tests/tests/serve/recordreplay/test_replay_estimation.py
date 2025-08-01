# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math
import statistics

import numpy as np
from max.serve.recordreplay import replay


def test_tnmean() -> None:
    tnmean = replay._tnmean
    inputs = np.arange(-10, 11)
    expected_outputs = np.array(
        [
            7.694598626706419e-23,
            1.0279773571668915e-18,
            5.052271083536895e-15,
            9.134720408376284e-12,
            6.075882855817676e-9,
            1.4867199409049054e-6,
            0.00013383446446857514,
            0.004437839042125663,
            0.055247862678989956,
            0.28759997093917833,
            0.7978845608028653,
            1.525135276160981,
            2.3732155328228397,
            3.2830986549304355,
            4.22560714448947,
            5.186503967125842,
            6.1584826045445995,
            7.137545613226504,
            8.12136811223611,
            9.10852310500287,
            10.09809323396251,
        ]
    )
    actual_outputs = np.vectorize(tnmean)(inputs)
    np.testing.assert_allclose(
        actual_outputs, expected_outputs, rtol=1e-6, atol=1e-6
    )
    far_inputs = np.arange(0, 220, 20)
    expected_far_outputs = np.array(
        [
            0.7978845608028653,
            20.04975306852785,
            40.02496884720726,
            60.016657420241124,
            80.01249609679823,
            100.00999800099926,
            120.0083321763276,
            140.00714212846574,
            160.0062495118141,
            180.00555521267296,
            200.00499975003126,
        ]
    )
    actual_far_outputs = np.vectorize(tnmean)(far_inputs)
    np.testing.assert_allclose(
        actual_far_outputs, expected_far_outputs, rtol=1e-6, atol=1e-6
    )


def test_low_truncated_expectation() -> None:
    low_truncated_expectation = replay._low_truncated_expectation
    # Reference points: Value of mean(truncated(Normal(X, Y), Z, Inf))
    # as evaluated by Distributions.jl.
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(), 0),
        0.7978845608028653,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(0, 10), 0),
        7.978845608028653,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(5, 10), 0),
        10.091604338370335,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(), -1),
        0.28759997093917833,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(0, 10), -1),
        7.353317485057806,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(5, 10), -1),
        9.591471366142866,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(), 2),
        2.3732155328228397,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(0, 10), 2),
        9.294158480856426,
    )
    assert np.isclose(
        low_truncated_expectation(statistics.NormalDist(5, 10), 2),
        11.172208536127343,
    )


def test_snapshot_eta() -> None:
    # Scenario: We have planned transactions but don't have any workers yet.
    # We have no idea what the ETA should be.
    assert math.isnan(
        replay.ProgressSnapshot(
            completed_transactions=0,
            transactions_in_progress=0,
            concurrency=0,
            total_transactions=1000,
            latency_distribution=statistics.NormalDist(math.nan, math.nan),
            completed_transactions_total_seconds=0,
            in_progress_transaction_durations=[],
            elapsed_seconds=0,
        ).estimated_seconds_remaining
    )
    # Scenario: A single transaction is in progress (50 seconds and going), and
    # none have finished.  There are 1000 after this.  Our best guess is that
    # we'll take at least as long as this one did.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=0,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(math.nan, math.nan),
            completed_transactions_total_seconds=0,
            in_progress_transaction_durations=[50],
            elapsed_seconds=50,
        ).estimated_seconds_remaining,
        # Arguably should be 50 * 1000, but then if there are none left we
        # guess 0 which is absurd.  Arguably we should not provide an estimate
        # at all in this case but this is better than nothing.
        50 * 1001,
        rtol=0,
        atol=1,
    )
    # Scenario: 1000 transactions all completed serially, all taking very close
    # to 1000 seconds each to complete.  There is 1 transaction left that has
    # just started.  We expect it to take just about 1000 seconds to complete.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(1000, 1),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[0],
            elapsed_seconds=1_000_000,
        ).estimated_seconds_remaining,
        1000,
        rtol=0,
        atol=1,
    )
    # Scenario: 1000 transactions all completed serially, all taking very close
    # to 1000 seconds each to complete.  There is 1 transaction left that is
    # 500 seconds in.  We expect it to take just about 500 seconds to complete.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(1000, 1),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[500],
            elapsed_seconds=1_000_500,
        ).estimated_seconds_remaining,
        500,
        rtol=0,
        atol=1,
    )
    # Scenario: 1000 transactions all completed serially, all taking very close
    # to 1000 seconds each to complete.  There is 1 transaction left that is
    # 1005 seconds in.  We expect it to finish imminently.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(1000, 1),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[1005],
            elapsed_seconds=1_001_005,
        ).estimated_seconds_remaining,
        0,
        rtol=0,
        atol=1,
    )
    # Same as previous scenario, but the last transaction is even more overdue,
    # by 200 seconds instead of 5 seconds.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(1000, 1),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[1200],
            elapsed_seconds=1_001_200,
        ).estimated_seconds_remaining,
        0,
        rtol=0,
        atol=1,
    )
    # Scenario: 1000 transactions all completed serially, with a large variance
    # in completion times but a mean of 1000 seconds to complete.  There is 1
    # transaction left that is 500 seconds in.  The average time across all
    # transactions is 1000 seconds, but excluding the transactions that took
    # under 500 seconds, the average time of the remaining transactions is
    # 1143.8 seconds, so guess 643.8 seconds left (slightly more accurate than
    # the naive assumption of 500 seconds left).
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(1000, 500),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[500],
            elapsed_seconds=1_000_500,
        ).estimated_seconds_remaining,
        643.8,
        rtol=0,
        atol=1,
    )
    # Same as last scenario but we're 1000 seconds in.  A more naive estimate
    # might say we're done now, but clearly we aren't.  The standard deviation
    # tells us we should expect another 399 seconds on average.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(1000, 500),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[1000],
            elapsed_seconds=1_001_000,
        ).estimated_seconds_remaining,
        398.9,
        rtol=0,
        atol=1,
    )
    # Same as last scenario but we're 1500 seconds in.  A more naive estimate
    # might say we expect to be done 500 seconds ago, or possibly imminently,
    # but both of those estimates are completely bogus.  This is where our
    # estimation scheme really shines over the naive scheme.  The standard
    # deviation tells us we should expect another 263 seconds on average.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=1,
            concurrency=1,
            total_transactions=1001,
            latency_distribution=statistics.NormalDist(1000, 500),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[1500],
            elapsed_seconds=1_001_500,
        ).estimated_seconds_remaining,
        262.6,
        rtol=0,
        atol=1,
    )
    # Scenario: 1000 transactions all completed serially, all taking very close
    # to 1000 seconds each to complete.  There are 1000 transactions left, none
    # of which have started yet.  We expect another million seconds to complete.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=0,
            concurrency=1,
            total_transactions=2000,
            latency_distribution=statistics.NormalDist(1000, 1),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[],
            elapsed_seconds=1_000_000,
        ).estimated_seconds_remaining,
        1_000_000,
        rtol=0,
        atol=10,
    )
    # Scenario: 1000 transactions all completed in parallel, all taking very close
    # to 1000 seconds each to complete.  There are 1000 transactions left, none
    # of which have started yet.  We expect about 1000 seconds to complete.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=0,
            concurrency=1000,
            total_transactions=2000,
            latency_distribution=statistics.NormalDist(1000, 1),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[],
            elapsed_seconds=1000,
        ).estimated_seconds_remaining,
        1000,
        rtol=0,
        atol=1,
    )
    # Scenario: 1000 transactions all completed in parallel, all taking very close
    # to 1000 seconds each to complete.  There are 1001 transactions left, none
    # of which have started yet.  We expect about 2000 seconds to complete.
    assert np.isclose(
        replay.ProgressSnapshot(
            completed_transactions=1000,
            transactions_in_progress=0,
            concurrency=1000,
            total_transactions=2001,
            latency_distribution=statistics.NormalDist(1000, 1),
            completed_transactions_total_seconds=1_000_000,
            in_progress_transaction_durations=[],
            elapsed_seconds=1000,
        ).estimated_seconds_remaining,
        2000,
        rtol=0,
        atol=1,
    )
