# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import defaultdict

from SDK.examples.di.load_balancer import LoadBalancer


def test_load_balancer() -> None:
    workers = ["a", "b", "c", "d"]
    load_balancer = LoadBalancer(workers)

    counter: dict[str, int] = defaultdict(int)
    for _ in range(20):
        counter[load_balancer.pick_worker()] += 1

    assert counter["a"] == 5
    assert counter["b"] == 5
    assert counter["c"] == 5
    assert counter["d"] == 5

    load_balancer.release("b")
    load_balancer.release("b")
    load_balancer.release("b")
    assert load_balancer.pick_worker() == "b"
    load_balancer.release("a")
    load_balancer.release("a")
    load_balancer.release("a")
    assert load_balancer.pick_worker() == "a"
